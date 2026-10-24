"""System 1 router: schema, skill, executor, service, and HTTP contract.

Everything here runs without torch, a camera, or a VLM server: the "VLM" is a
`ScriptedVLMClient` that returns canned JSON, and the encoder is a
deterministic fake. The real Qwen3.8-27B brain is exercised on the robot.
"""

import json
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from bebop_vision.places import PlaceStore, PlacesService, StaticCapture
from bebop_vision.places import View as PlaceView
from bebop_vision.places.submap import encode_rle, rle_true_count
from bebop_vision.router import (
    CAPTURE_MAP,
    DEFER,
    REFINE_MAP,
    REJECT_CAPTURE,
    REMEMBER_PLACE,
    SKIP,
    UPDATE_PLACE,
    CaptureMapExecutor,
    CaptureMapSkill,
    DecisionParseError,
    PlaceMemoryExecutor,
    PlaceMemorySkill,
    RouterService,
    ScriptedVLMClient,
    View,
    VoiceBusyProbe,
    build_messages,
    extract_json_object,
    normalize_decision,
    parse_decision,
)
from bebop_vision.router.schema import Candidate, Observation, SkillDecision


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeEncoder:
    dim = 4
    device = "fake"

    def __init__(self, table=None):
        self.table = dict(table or {})

    @property
    def loaded(self):
        return True

    def encode_bgr(self, bgr):  # pragma: no cover - unused
        raise NotImplementedError

    def encode_jpeg(self, jpeg):
        if jpeg in self.table:
            v = np.asarray(self.table[jpeg], np.float32)
        else:
            v = np.asarray([float(sum(jpeg) % 997), 1.0, 0.0, 0.0], np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def _reply(**kwargs) -> str:
    return json.dumps(kwargs)


def _map_snapshot(
    *,
    occupied=None,
    free=None,
    width=4,
    height=3,
    origin=(-0.5, -0.4),
    resolution=0.05,
    pose=(1.0, 2.0, 10.0),
    pose_valid=True,
    gravity_valid=True,
    walls=None,
    pose_delta_m=None,
    revision=1,
):
    """A valid :9096 /structure snapshot for capture tests."""
    occ = (np.zeros((height, width), dtype=bool) if occupied is None
           else np.asarray(occupied, dtype=bool))
    fr = (np.zeros((height, width), dtype=bool) if free is None
          else np.asarray(free, dtype=bool))
    body = {
        "frame_id": "map",
        "gravity": {
            "valid": gravity_valid,
            "up": [0.0, -1.0, 0.0] if gravity_valid else None,
            "roll_deg": 0.0 if gravity_valid else None,
            "pitch_deg": 1.0 if gravity_valid else None,
        },
        "floor": {
            "valid": gravity_valid,
            "normal": [0.0, -1.0, 0.0] if gravity_valid else None,
            "offset_m": 1.3 if gravity_valid else None,
            "camera_height_m": 1.3 if gravity_valid else None,
        },
        "grid": {
            "origin": [float(origin[0]), float(origin[1])],
            "resolution_m": float(resolution),
            "width": int(width),
            "height": int(height),
            "occupied_count": int(occ.sum()),
            "free_count": int(fr.sum()),
            "encoding": "rle_base64",
            "occupied_rle": encode_rle(occ),
            "free_rle": encode_rle(fr),
        },
        "walls": [] if walls is None else walls,
        "wall_cells": [],
        "pose": {
            "valid": pose_valid,
            "x": float(pose[0]) if pose_valid else None,
            "y": float(pose[1]) if pose_valid else None,
            "yaw_deg": float(pose[2]) if pose_valid else None,
            "t_ns": 123,
            "segment": 0,
        },
        "revision": revision,
    }
    if pose_delta_m is not None:
        body["pose_delta_m"] = float(pose_delta_m)
    return body


def _recognizer(place, *, matched=True, score=0.9, margin=0.4):
    """A /whereami-shaped recognizer (no camera or encoder needed)."""

    def recognize(_views):
        return {
            "matched": matched and place is not None,
            "place": ({"id": place.id, "name": place.name}
                      if matched and place is not None else None),
            "score": score,
            "margin": margin,
            "min_score": 0.5,
            "min_margin": 0.05,
        }

    return recognize


@pytest.fixture()
def store(tmp_path):
    s = PlaceStore(tmp_path / "places")
    yield s
    s.close()


class Clock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t


# --- schema -----------------------------------------------------------------


def test_extract_json_object_from_prose_and_fence():
    text = 'Sure!\n```json\n{"decision": "skip", "confidence": 0.4}\n```\n'
    assert extract_json_object(text)["decision"] == "skip"


def test_extract_json_object_rejects_garbage():
    with pytest.raises(DecisionParseError):
        extract_json_object("no object here")


def test_extract_json_object_handles_braces_in_strings():
    text = '{"decision": "remember_place", "arguments": {"name": "a {weird} room"}}'
    assert extract_json_object(text)["arguments"]["name"] == "a {weird} room"


def test_normalize_decision_aliases():
    assert normalize_decision("create_place") == REMEMBER_PLACE
    assert normalize_decision("Enrich") == UPDATE_PLACE
    assert normalize_decision("wait") == DEFER
    assert normalize_decision("nonsense") is None


def test_parse_decision_normalises_and_clamps():
    d = parse_decision(
        '{"decision": "create_place", "confidence": 3.0, '
        '"name": "Workshop", "observations": ["a bench"]}'
    )
    assert d.decision == REMEMBER_PLACE
    assert d.confidence == 1.0
    assert d.arguments["name"] == "Workshop"  # flat field lifted into arguments
    assert d.observations == ["a bench"]


def test_parse_decision_keeps_unknown_token_for_sanitiser():
    d = parse_decision('{"decision": "teleport", "confidence": 0.9}')
    assert d.decision == "teleport"


# --- prompt -----------------------------------------------------------------


def test_build_messages_includes_images_and_candidates():
    obs = Observation(
        views=[View("far", _jpeg(10), 1)],
        candidates=[Candidate("p1", "Kitchen", 0.81, facts=("sink",), view_count=2)],
        instruction="remember this as the workshop",
        places_known=1,
    )
    msgs = build_messages(obs)
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    text = " ".join(p["text"] for p in content if p["type"] == "text")
    assert "p1" in text and "Kitchen" in text
    assert "workshop" in text
    images = [p for p in content if p["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# --- skill: deciding --------------------------------------------------------


def _skill(store, encoder, replies, **kw):
    vlm = ScriptedVLMClient(replies)
    return PlaceMemorySkill(store, encoder, vlm, **kw), vlm


def test_assess_remembers_unfamiliar_room(store):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", reason_code="new_room",
        confidence=0.8, arguments={"name": "Workshop"},
        observations=["a workbench under a window"],
    ))
    out = skill.assess([View("far", a, 1)])
    assert out.decision.decision == REMEMBER_PLACE
    assert out.decision.arguments["name"] == "Workshop"
    assert out.candidates == []


def test_assess_retrieves_candidates(store):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [1, 0, 0, 0]})
    store.create_place("Kitchen")
    p = store.get_place("Kitchen")
    store.add_descriptor(p.id, np.array([1.0, 0, 0, 0]))

    skill, _ = _skill(store, enc, _reply(decision="skip"))
    out = skill.assess([View("far", b, 1)])
    assert [c.name for c in out.candidates] == ["Kitchen"]
    assert out.candidates[0].score == pytest.approx(1.0, abs=1e-6)


def test_assess_defers_on_vlm_error(store):
    a = _jpeg(20)

    class Broken:
        describe = "broken"

        def complete(self, *a, **k):
            raise RuntimeError("connection refused")

    skill = PlaceMemorySkill(store, FakeEncoder(), Broken())
    out = skill.assess([View("far", a, 1)])
    assert out.decision.decision == DEFER
    assert out.decision.reason_code == "vlm_error"
    assert out.error and "refused" in out.error


def test_assess_defers_on_unparseable_output(store):
    a = _jpeg(20)
    skill, _ = _skill(store, FakeEncoder(), "I think it is a nice room, yes.")
    out = skill.assess([View("far", a, 1)])
    assert out.decision.decision == DEFER
    assert out.decision.reason_code == "unparseable_output"


def test_sanitize_rejects_unknown_decision_and_missing_name(store):
    a = _jpeg(20)
    enc = FakeEncoder()
    skill, _ = _skill(store, enc, _reply(decision="teleport"))
    out = skill.assess([View("far", a, 1)])
    assert out.decision.reason_code == "unknown_decision"

    skill, _ = _skill(store, enc, _reply(decision="remember_place"))
    out = skill.assess([View("far", a, 1)])
    assert out.decision.reason_code == "missing_name"


def test_sanitize_requires_evidence(store):
    enc = FakeEncoder()
    skill, _ = _skill(store, enc, _reply(decision="skip"), min_evidence_views=1)
    out = skill.assess([])
    # no views -> at least one frame required
    assert out.decision.decision == DEFER
    assert out.decision.reason_code == "insufficient_evidence"


def test_sanitize_update_resolves_candidate_only(store):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [1, 0, 0, 0]})
    p = store.create_place("Kitchen")
    store.add_descriptor(p.id, np.array([1.0, 0, 0, 0]))

    skill, _ = _skill(store, enc, _reply(
        decision="update_place", arguments={"place_id": "does-not-exist"}))
    out = skill.assess([View("far", b, 1)])
    assert out.decision.decision == DEFER
    assert out.decision.reason_code == "unknown_place"

    skill, _ = _skill(store, enc, _reply(
        decision="update_place", arguments={"place_id": p.id}))
    out = skill.assess([View("far", b, 1)])
    assert out.decision.decision == UPDATE_PLACE
    assert out.decision.arguments["place_id"] == p.id
    assert out.decision.arguments["place_name"] == "Kitchen"


# --- executor: acting -------------------------------------------------------


def _assess(skill, views, **kw):
    return skill.assess(views, **kw).decision


def test_executor_creates_and_dedupes_within_cooldown(store):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [1, 0, 0, 0]})
    clock = Clock(1000.0)
    ex = PlaceMemoryExecutor(store, enc, cooldown_s=180.0, clock=clock)

    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", reason_code="new_room",
        arguments={"name": "Workshop"}, observations=["a workbench"]))
    d1 = _assess(skill, [View("far", a, 1)], instruction="remember the workshop")
    out1 = ex.apply(d1, [View("far", a, 1)], instruction="remember the workshop")
    assert out1["applied"] is True and out1["action"] == REMEMBER_PLACE
    pid = out1["place_id"]
    assert store.count() == 1
    assert store.descriptor_count(pid) == 1
    place = store.get_place(pid)
    assert any("workbench" in f.text for f in place.facts)
    assert any(f.source == "operator" for f in place.facts)

    # A second remember moments later folds into the place just created.
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", arguments={"name": "Workshop"}))
    d2 = _assess(skill, [View("far", b, 2)])
    out2 = ex.apply(d2, [View("far", b, 2)])
    assert out2["action"] == UPDATE_PLACE
    assert out2["reason_code"] == "cooldown_dedupe"
    assert out2["place_id"] == pid
    assert store.count() == 1
    assert store.descriptor_count(pid) == 2

    # After the cooldown a genuinely new room creates a second place.
    clock.t = 1000.0 + 200.0
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", arguments={"name": "Bathroom"}))
    d3 = _assess(skill, [View("far", b, 3)])
    out3 = ex.apply(d3, [View("far", b, 3)])
    assert out3["action"] == REMEMBER_PLACE
    assert store.count() == 2


def test_executor_dedupes_by_existing_name(store):
    a = _jpeg(20)
    enc = FakeEncoder()
    ex = PlaceMemoryExecutor(store, enc)
    store.create_place("Kitchen")
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", arguments={"name": "kitchen"}))
    d = _assess(skill, [View("far", a, 1)])
    out = ex.apply(d, [View("far", a, 1)])
    assert out["action"] == UPDATE_PLACE
    assert out["reason_code"] == "deduped_by_name"
    assert store.count() == 1


def test_executor_caps_evidence_and_facts(store):
    a, b, c = _jpeg(20), _jpeg(80), _jpeg(140)
    enc = FakeEncoder()
    ex = PlaceMemoryExecutor(store, enc, max_views_per_call=2,
                             max_facts_per_call=1)
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", arguments={"name": "Hall"},
        observations=["a door", "a coat rack", "a rug"]))
    views = [View("far", a, 1), View("near", b, 2), View("far", c, 3)]
    d = _assess(skill, views)
    out = ex.apply(d, views)
    assert out["keyframes"] == 2
    assert out["facts"] == 1


def test_executor_ignores_write_failures(store):
    class BoomStore:
        def get_place(self, *_a, **_k):
            return None

    ex = PlaceMemoryExecutor(BoomStore(), None)
    from bebop_vision.router.schema import SkillDecision

    out = ex.apply(
        SkillDecision(decision=UPDATE_PLACE, arguments={"place_id": "x"}),
        [View("far", _jpeg(1), 1)],
    )
    assert out["applied"] is False and out["reason_code"] == "unknown_place"


def test_executor_skip_and_defer_are_noops(store):
    ex = PlaceMemoryExecutor(store, None)
    for name in (SKIP, DEFER):
        from bebop_vision.router.schema import SkillDecision

        out = ex.apply(SkillDecision(decision=name), [])
        assert out["applied"] is False and out["action"] == name


# --- service ----------------------------------------------------------------


def _service(store, encoder, views, replies, **kw):
    skill, _ = _skill(store, encoder, replies)
    ex = PlaceMemoryExecutor(store, encoder, cooldown_s=kw.pop("cooldown_s", 0.0))
    svc = RouterService(skill, ex, StaticCapture(views), interval_s=0.05, **kw)
    return svc


def test_service_defers_without_frames(store):
    svc = _service(store, FakeEncoder(), [], _reply(decision="skip"))
    out = svc.assess_once()
    assert out["ok"] is False and out["reason"] == "no_frames"
    assert svc.decisions()[0]["reason_code"] == "no_frames"


def test_service_applies_and_logs(store):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    svc = _service(store, enc, [PlaceView("far", a, 1)], _reply(
        decision="remember_place", reason_code="new_room",
        arguments={"name": "Workshop"}, observations=["a bench"]))
    out = svc.assess_once(source="http")
    assert out["ok"] is True and out["applied"]["applied"] is True
    assert store.count() == 1
    rec = svc.decisions()[0]
    assert rec["source"] == "http" and rec["applied"] is True
    assert svc.status()["assessments"] == 1
    assert svc.status()["applied"] == 1
    assert any("remember_place" in e for e in svc.events())


def test_service_loop_runs_and_stops(store, tmp_path):
    import time

    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    skill, _ = _skill(store, enc, _reply(decision="skip", reason_code="seen"))
    ex = PlaceMemoryExecutor(store, enc, cooldown_s=0.0)
    svc = RouterService(skill, ex, StaticCapture([PlaceView("far", a, 1)]),
                        interval_s=0.02, initial_delay_s=0.01)
    svc.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and svc.status()["assessments"] < 2:
        time.sleep(0.02)
    svc.stop()
    assert svc.status()["assessments"] >= 2
    assert svc.status()["running"] is False


# --- map-capture skill ------------------------------------------------------


def _capture_skill(store, replies, *, snapshot=None, recognizer=None, **kw):
    vlm = ScriptedVLMClient(replies)
    if snapshot is None:
        snapshot = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    fetch = snapshot if callable(snapshot) else (lambda: snapshot)
    kw.setdefault("motion_window_s", 0.0)
    kw.setdefault("min_cells", 1)
    kw.setdefault("clock", Clock())
    skill = CaptureMapSkill(
        store, FakeEncoder(), vlm, structure_fetch=fetch,
        recognizer=recognizer, **kw,
    )
    return skill, vlm


def test_capture_gate_failures_defer_after_asking_the_vlm(store):
    p = store.create_place("Lab")
    views = [View("far", _jpeg(20), 1)]
    cases = [
        ("gravity_invalid", "gate_gravity_invalid",
         _map_snapshot(gravity_valid=False), None),
        ("pose_invalid", "gate_pose_invalid",
         _map_snapshot(pose_valid=False), None),
        ("robot_moving", "gate_moving", _map_snapshot(
            pose_delta_m=1.0, free=np.ones((3, 4), dtype=bool)), None),
        ("too_few_cells", "gate_too_few_cells", _map_snapshot(), None),
        ("not_recognized_here", "gate_not_recognized_here",
         _map_snapshot(free=np.ones((3, 4), dtype=bool)),
         _recognizer(p, matched=False)),
    ]
    for gate_reason, expected, snap, recognizer in cases:
        skill, vlm = _capture_skill(
            store, _reply(decision="capture_map"),
            snapshot=snap,
            recognizer=recognizer or _recognizer(p),
            min_cells=1,
        )
        out = skill.assess(views)
        # The model is prompted every tick, even with a closed gate.
        assert len(vlm.calls) == 1
        assert out.decision.decision == DEFER
        assert out.decision.reason_code == expected
        assert out.decision.arguments["gate_reason"] == gate_reason
        ex = CaptureMapExecutor(store, structure_fetch=lambda: snap,
                                cooldown_s=0.0)
        assert ex.apply(out.decision)["written"] is False
        assert store.get_place_map(p.id) is None


def test_capture_target_must_be_the_recognized_place(store):
    here = store.create_place("Lab")
    elsewhere = store.create_place("Kitchen")
    skill, vlm = _capture_skill(
        store, _reply(decision="capture_map"),
        recognizer=_recognizer(here), min_cells=1,
    )
    out = skill.assess([View("far", _jpeg(20), 1)],
                       target_place_id=elsewhere.id)
    assert out.decision.reason_code == "gate_not_recognized_here"
    assert out.decision.arguments["gate_reason"] == "not_recognized_here"
    assert len(vlm.calls) == 1
    assert store.get_place_map(here.id) is None


def test_capture_map_applies_via_the_executor(store):
    p = store.create_place("Lab")
    occ = np.zeros((3, 4), dtype=bool)
    occ[1, 1] = True
    snap = _map_snapshot(occupied=occ)
    skill, vlm = _capture_skill(
        store,
        _reply(decision="capture_map", reason_code="new_map",
               confidence=0.9, arguments={"place_id": p.id},
               observations=["a lab bench"]),
        snapshot=snap, recognizer=_recognizer(p),
    )
    out = skill.assess([View("far", _jpeg(20), 1)])
    assert out.context is not None and out.context.gate_ok is True
    assert out.context.place_id == p.id
    assert out.decision.decision == CAPTURE_MAP
    assert len(vlm.calls) == 1

    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap, cooldown_s=0.0)
    applied = ex.apply(out.decision)
    assert applied["applied"] is True
    assert applied["created"] is True and applied["written"] is True
    place_map = store.get_place_map(p.id)
    assert place_map is not None and place_map.revision == 1
    assert rle_true_count(place_map.occupied_rle) == 1
    assert applied["revision"] == 1


def test_reject_capture_writes_nothing(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    skill, _ = _capture_skill(
        store, _reply(decision="reject_capture", reason_code="thin_map",
                      arguments={"place_id": p.id}),
        snapshot=snap, recognizer=_recognizer(p),
    )
    out = skill.assess([View("far", _jpeg(20), 1)])
    assert out.decision.decision == REJECT_CAPTURE
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap, cooldown_s=0.0)
    applied = ex.apply(out.decision)
    assert applied["applied"] is False
    assert applied["written"] is False
    assert store.get_place_map(p.id) is None


def test_unknown_capture_decision_defers_and_writes_nothing(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    skill, _ = _capture_skill(
        store, _reply(decision="teleport", arguments={"place_id": p.id}),
        snapshot=snap, recognizer=_recognizer(p),
    )
    out = skill.assess([View("far", _jpeg(20), 1)])
    assert out.decision.decision == DEFER
    assert out.decision.reason_code == "unknown_decision"
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap, cooldown_s=0.0)
    assert ex.apply(out.decision)["written"] is False
    assert store.get_place_map(p.id) is None


def test_executor_cooldown_suppresses_a_repeat_capture(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    clock = Clock(1000.0)
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap,
                            cooldown_s=180.0, clock=clock)
    decision = SkillDecision(decision=CAPTURE_MAP,
                             arguments={"place_id": p.id})
    first = ex.apply(decision)
    assert first["applied"] is True and first["revision"] == 1

    second = ex.apply(decision)
    assert second["applied"] is False
    assert second["reason_code"] == "cooldown"
    assert store.get_place_map(p.id).revision == 1

    clock.t += 181.0
    third = ex.apply(decision)
    assert third["applied"] is True
    assert store.get_place_map(p.id).revision == 2


def test_executor_ignores_geometry_authored_by_the_model(store):
    p = store.create_place("Lab")
    occ = np.zeros((3, 4), dtype=bool)
    occ[2, 3] = True
    snap = _map_snapshot(occupied=occ)
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap, cooldown_s=0.0)
    injected = SkillDecision(
        decision=CAPTURE_MAP,
        arguments={
            "place_id": p.id,
            # A hostile/hallucinated payload must be ignored completely.
            "grid": {"width": 99, "height": 99, "occupied_rle": "bogus"},
            "occupied_rle": "bogus",
            "occupied_count": 999,
            "walls": [[9.0, 9.0, 9.0, 9.0]],
        },
    )
    out = ex.apply(injected)
    assert out["applied"] is True
    place_map = store.get_place_map(p.id)
    assert (place_map.width, place_map.height) == (4, 3)
    assert rle_true_count(place_map.occupied_rle) == 1
    assert place_map.walls == []


def test_capture_is_assessed_on_every_service_tick(store):
    svc, lab, place_vlm, cap_vlm = _multi_service(
        store,
        _reply(decision="skip", reason_code="seen"),
        _reply(decision="skip", reason_code="nothing_new"),
    )
    for _ in range(3):
        out = svc.tick()
        assert out["ok"] is True
        assert {result["skill"] for result in out["results"]} == {
            "place_memory", "capture_map"}
    # There is no event-trigger skip path: both brains were asked every tick.
    assert len(place_vlm.calls) == 3
    assert len(cap_vlm.calls) == 3
    assert store.get_place_map(lab.id) is None
    assert not hasattr(svc.skills[1].skill, "event_trigger")

    status = svc.status()
    capture = next(s for s in status["skills"] if s["name"] == "capture_map")
    assert capture["event_gated"] is False
    assert capture["last_trigger"] is None
    assert capture["gate"]["evaluated"] is True


def test_capture_loop_prompts_the_vlm_every_tick(store):
    import time

    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    skill, vlm = _capture_skill(
        store, _reply(decision="skip", reason_code="nothing_new"),
        snapshot=snap, recognizer=_recognizer(p),
    )
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap,
                            cooldown_s=180.0)
    svc = RouterService(
        skill, ex, StaticCapture([PlaceView("far", _jpeg(20), 1)]),
        interval_s=0.02, initial_delay_s=0.01,
    )
    svc.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(vlm.calls) < 2:
        time.sleep(0.02)
    svc.stop()
    assert len(vlm.calls) >= 2  # no event-trigger skip path


def test_gate_failure_is_recorded_and_writes_nothing(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(pose_valid=False,
                         free=np.ones((3, 4), dtype=bool))
    skill, vlm = _capture_skill(
        store, _reply(decision="capture_map"), snapshot=snap,
        recognizer=_recognizer(p), min_cells=1,
    )
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap, cooldown_s=0.0)
    svc = RouterService(
        skill, ex, StaticCapture([PlaceView("far", _jpeg(20), 1)]))

    out = svc.tick()
    assert len(vlm.calls) == 1          # asked even though the gate is closed
    rec = out["record"]
    assert rec["decision"] == DEFER
    assert rec["reason_code"] == "gate_pose_invalid"
    assert rec["gate_reason"] == "pose_invalid"
    assert rec["applied"] is False
    assert store.get_place_map(p.id) is None
    assert svc.decisions()[0]["gate_reason"] == "pose_invalid"


def test_service_tick_skips_voice_and_cooldown_blocks_repeat_writes(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    clock = Clock(1000.0)
    skill, vlm = _capture_skill(
        store, _reply(decision="capture_map"), snapshot=snap,
        recognizer=_recognizer(p), clock=clock,
    )
    ex = CaptureMapExecutor(store, structure_fetch=lambda: snap,
                            cooldown_s=180.0, clock=clock)
    busy = {"value": True}
    svc = RouterService(
        skill, ex, StaticCapture([PlaceView("far", _jpeg(20), 1)]),
        voice_busy=lambda: busy["value"],
    )

    out = svc.tick()
    assert out["skipped"] == "voice_busy"
    assert vlm.calls == []

    busy["value"] = False
    first = svc.tick()
    assert first["ok"] is True
    assert first["applied"]["action"] == CAPTURE_MAP
    assert store.get_place_map(p.id).revision == 1
    assert len(vlm.calls) == 1

    # The brain is still asked every tick; the executor cooldown suppresses
    # the second write even though the model keeps saying capture_map.
    second = svc.tick()
    assert second["ok"] is True
    assert second["record"]["applied"] is False
    assert second["applied"]["reason_code"] == "cooldown"
    assert store.get_place_map(p.id).revision == 1
    assert len(vlm.calls) == 2
    assert skill.gate_status()["in_cooldown"] is True

    clock.t += 181.0
    third = svc.tick()
    assert third["record"]["applied"] is True
    assert store.get_place_map(p.id).revision == 2
    assert len(vlm.calls) == 3





# --- multi-skill wiring -----------------------------------------------------


def _multi_service(store, place_reply, capture_reply, *, snapshot=None,
                   voice_busy=None, capture_kw=None):
    lab = store.create_place("Lab")
    snap = snapshot or _map_snapshot(free=np.ones((3, 4), dtype=bool))
    place_skill, place_vlm = _skill(store, FakeEncoder(), place_reply)
    cap_skill, cap_vlm = _capture_skill(
        store, capture_reply, snapshot=snap, recognizer=_recognizer(lab),
        place_stable_s=(capture_kw or {}).get("place_stable_s", 0.0),
        min_cells_gained=(capture_kw or {}).get("min_cells_gained", 100),
        cooldown_s=(capture_kw or {}).get("cooldown_s", 0.0),
    )
    place_ex = PlaceMemoryExecutor(store, FakeEncoder(), cooldown_s=0.0)
    cap_ex = CaptureMapExecutor(store, structure_fetch=lambda: snap,
                                cooldown_s=0.0)
    svc = RouterService(
        place_skill, place_ex,
        capture=StaticCapture([PlaceView("far", _jpeg(20), 1)]),
        skills=[(cap_skill, cap_ex)],
        voice_busy=voice_busy,
    )
    return svc, lab, place_vlm, cap_vlm


def test_two_registrations_are_both_evaluated(store):
    svc, lab, place_vlm, cap_vlm = _multi_service(
        store,
        _reply(decision="remember_place", arguments={"name": "Workshop"}),
        _reply(decision="capture_map"),
    )
    out = svc.tick()
    assert out["ok"] is True
    assert {result["skill"] for result in out["results"]} == {
        "place_memory", "capture_map"}
    assert len(place_vlm.calls) == 1
    assert len(cap_vlm.calls) == 1
    assert store.get_place("Workshop") is not None
    assert store.get_place_map(lab.id) is not None
    assert svc.status()["registrations"] == 2
    assert [s["name"] for s in svc.status()["skills"]] == [
        "place_memory", "capture_map"]


def test_multi_skill_tick_skips_off_while_voice_busy(store):
    svc, _lab, place_vlm, cap_vlm = _multi_service(
        store,
        _reply(decision="remember_place", arguments={"name": "Workshop"}),
        _reply(decision="capture_map", arguments={"place_id": "ignored"}),
        voice_busy=lambda: True,
    )
    assert svc.tick() == {"ok": False, "skipped": "voice_busy"}
    assert place_vlm.calls == [] and cap_vlm.calls == []


def test_single_registration_constructor_is_backward_compatible(store):
    a = _jpeg(20)
    skill, vlm = _skill(store, FakeEncoder({a: [1, 0, 0, 0]}),
                        _reply(decision="skip"))
    executor = PlaceMemoryExecutor(store, FakeEncoder())
    svc = RouterService(skill, executor, StaticCapture([PlaceView("far", a, 1)]))
    assert len(svc.registrations) == 1
    assert svc.skill is skill and svc.executor is executor
    out = svc.tick()
    assert out["ok"] is True
    assert len(vlm.calls) == 1
    assert svc.status()["registrations"] == 1


def test_voice_busy_probe_uses_turn_signal_and_fails_closed():
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"body": {"phase": "ready"}, "delay": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            import time as _t
            _t.sleep(state["delay"])
            data = json.dumps(state["body"]).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the probe gave up (the timeout case)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/healthz"
    try:
        probe = VoiceBusyProbe(url, timeout=0.5)
        assert probe() is True           # up, but no signal -> fail closed
        state["body"] = {"turn_active": True, "turns_active": 1}
        assert probe() is True
        state["body"] = {"turn_active": False, "turns_active": 0}
        assert probe() is False
        state["body"] = {"in_flight": 0}
        assert probe() is False
        state["body"] = {"busy": "false"}  # strings are not a busy signal
        assert probe() is False
        state["body"] = ["not", "an", "object"]
        assert probe() is True
        state["body"], state["delay"] = {"turn_active": False}, 0.5
        assert VoiceBusyProbe(url, timeout=0.1)() is True  # hung -> busy
        state["delay"] = 0.0
    finally:
        httpd.shutdown()
        httpd.server_close()

    # A stopped service (nothing listening) has no turn to collide with.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    free = sock.getsockname()[1]
    sock.close()
    dead = VoiceBusyProbe(f"http://127.0.0.1:{free}/healthz", timeout=0.2)
    assert dead() is False





def test_capture_gate_status_updates_on_every_assessment(store):
    p = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))
    clock = Clock(100.0)
    skill, vlm = _capture_skill(
        store, _reply(decision="skip", reason_code="seen"), snapshot=snap,
        recognizer=_recognizer(p), clock=clock, cooldown_s=180.0,
    )
    views = [View("far", _jpeg(20), 1)]

    # Every assess call prompts the brain and refreshes gate telemetry.
    assert skill.assess(views).decision.decision == SKIP
    gate = skill.gate_status()
    assert gate["evaluated"] is True
    assert gate["gate_ok"] is True
    assert gate["place_id"] == p.id
    assert gate["place_changed"] is True
    assert gate["stable_for_s"] == pytest.approx(0.0)
    assert gate["place_stable_s"] == pytest.approx(10.0)
    assert gate["cells_gained"] == 12
    assert gate["min_cells_gained"] == 100
    assert gate["in_cooldown"] is False
    assert "trigger" not in gate  # event triggers are gone
    assert len(vlm.calls) == 1

    clock.t = 120.0
    assert skill.assess(views).decision.decision == SKIP
    gate = skill.gate_status()
    assert gate["stable_for_s"] == pytest.approx(20.0)
    assert gate["place_changed"] is False
    assert len(vlm.calls) == 2

    # Cooldown after a successful capture is visible without gating the prompt.
    skill.note_outcome({"applied": True, "place_id": p.id})
    clock.t = 130.0
    assert skill.assess(views).decision.decision == SKIP
    gate = skill.gate_status()
    assert gate["in_cooldown"] is True
    assert gate["cooldown_remaining_s"] == pytest.approx(170.0)
    assert len(vlm.calls) == 3


def test_router_status_exposes_each_skills_gate(store):
    svc, lab, _place_vlm, cap_vlm = _multi_service(
        store,
        _reply(decision="skip", reason_code="seen"),
        _reply(decision="capture_map"),
        capture_kw={"place_stable_s": 10.0},
    )
    out = svc.tick()
    assert out["results"][1]["ok"] is True
    assert len(cap_vlm.calls) == 1
    assert store.get_place_map(lab.id) is not None

    status = svc.status()
    json.dumps(status)  # /router/status must stay JSON-serializable
    capture = next(s for s in status["skills"] if s["name"] == "capture_map")
    assert capture["event_gated"] is False  # prompted every tick
    assert capture["last_trigger"] is None
    gate = capture["gate"]
    assert gate["evaluated"] is True
    assert gate["gate_ok"] is True
    assert gate["stable_for_s"] == pytest.approx(0.0)
    assert gate["place_stable_s"] == pytest.approx(10.0)
    assert "place_changed" in gate
    assert gate["cells_gained"] >= 0
    assert gate["min_cells_gained"] == 100
    assert gate["in_cooldown"] is False
    assert "trigger" not in gate

    place = next(s for s in status["skills"] if s["name"] == "place_memory")
    assert "gate" not in place  # only skills that expose one





# --- HTTP -------------------------------------------------------------------


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_http_router_endpoints(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store = PlaceStore(tmp_path / "places")
    skill, _ = _skill(store, enc, _reply(
        decision="remember_place", reason_code="new_room",
        arguments={"name": "Workshop"}, observations=["a bench"]))
    router = RouterService(skill, PlaceMemoryExecutor(store, enc, cooldown_s=0.0),
                           StaticCapture([PlaceView("far", a, 1)]))
    svc = PlacesService(store, capture=StaticCapture([]), encoder=enc,
                        host="127.0.0.1", port=0, router=router)
    svc.start()
    try:
        port = svc.bound_port
        _, health = _get(port, "/healthz")
        assert health["router"]["enabled"] is True

        status, assess = _post(port, "/router/assess", {})
        assert status == 200 and assess["ok"] is True
        assert assess["applied"]["action"] == REMEMBER_PLACE

        _, decisions = _get(port, "/router/decisions?limit=5")
        assert len(decisions["decisions"]) == 1

        _, status_body = _get(port, "/router/status")
        assert status_body["assessments"] == 1
    finally:
        svc.stop()
        store.close()


def test_http_router_assess_skill_target(tmp_path):
    a = _jpeg(20)
    store = PlaceStore(tmp_path / "places")
    lab = store.create_place("Lab")
    snap = _map_snapshot(free=np.ones((3, 4), dtype=bool))

    place_vlm = ScriptedVLMClient(_reply(decision="skip", reason_code="seen"))
    place_skill = PlaceMemorySkill(store, FakeEncoder(), place_vlm)
    cap_vlm = ScriptedVLMClient(_reply(decision="capture_map"))
    cap_skill = CaptureMapSkill(
        store, FakeEncoder(), cap_vlm, structure_fetch=lambda: snap,
        recognizer=_recognizer(lab), motion_window_s=0.0, min_cells=1,
    )
    router = RouterService(
        place_skill, PlaceMemoryExecutor(store, FakeEncoder(), cooldown_s=0.0),
        StaticCapture([PlaceView("far", a, 1)]),
        skills=[(cap_skill,
                 CaptureMapExecutor(store, structure_fetch=lambda: snap,
                                    cooldown_s=0.0))],
    )
    svc = PlacesService(store, capture=StaticCapture([]), encoder=FakeEncoder(),
                        host="127.0.0.1", port=0, router=router)
    svc.start()
    try:
        port = svc.bound_port

        # Select capture_map by body: place memory must not be called.
        status, out = _post(port, "/router/assess",
                            {"skill": "capture_map"})
        assert status == 200
        assert out["applied"]["action"] == CAPTURE_MAP
        assert out["record"]["skill"] == "capture_map"
        assert place_vlm.calls == []
        assert store.get_place_map(lab.id) is not None
        first_revision = store.get_place_map(lab.id).revision

        # And by query string.
        status, out = _post(port, "/router/assess?skill=capture_map", {})
        assert status == 200
        assert out["record"]["skill"] == "capture_map"
        assert store.get_place_map(lab.id).revision == first_revision + 1

        # An unknown skill is a 400 that lists what is registered.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(port, "/router/assess", {"skill": "nope"})
        assert exc.value.code == 400
        error = json.loads(exc.value.read())["error"]
        assert "place_memory" in error and "capture_map" in error

        # No skill target keeps the original primary-skill behaviour.
        status, out = _post(port, "/router/assess", {})
        assert status == 200
        assert out["record"]["skill"] == "place_memory"
        assert len(place_vlm.calls) == 1
    finally:
        svc.stop()
        store.close()


def test_http_router_disabled_is_503(tmp_path):
    store = PlaceStore(tmp_path / "places")
    svc = PlacesService(store, encoder=None, host="127.0.0.1", port=0)
    svc.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(svc.bound_port, "/router/status")
        assert exc.value.code == 503
    finally:
        svc.stop()
        store.close()


# --- VLM client -------------------------------------------------------------


def test_chat_vlm_client_roundtrip():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from bebop_vision.router import ChatVLMClient

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(n))
            payload = json.dumps(
                {"choices": [{"message": {"content": '{"decision": "skip"}'}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = ChatVLMClient(
            f"http://127.0.0.1:{httpd.server_address[1]}", model="m", timeout=5.0
        )
        out = client.complete([{"role": "user", "content": "hi"}], max_tokens=64)
        assert out == '{"decision": "skip"}'
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["max_tokens"] == 64
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_vlm_client_wraps_transport_errors():
    from bebop_vision.router import ChatVLMClient, VLMError

    client = ChatVLMClient("http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(VLMError):
        client.complete([{"role": "user", "content": "hi"}])

