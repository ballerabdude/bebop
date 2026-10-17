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
from bebop_vision.router import (
    DEFER,
    REMEMBER_PLACE,
    SKIP,
    UPDATE_PLACE,
    DecisionParseError,
    PlaceMemoryExecutor,
    PlaceMemorySkill,
    RouterService,
    ScriptedVLMClient,
    View,
    build_messages,
    extract_json_object,
    normalize_decision,
    parse_decision,
)
from bebop_vision.router.schema import Observation, Candidate


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

