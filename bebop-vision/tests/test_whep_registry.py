"""WHEP multi-viewer bookkeeping: viewer cap, shared encoder, DELETE,
last-viewer release, liveness. No cameras, no GStreamer: hubs, peers and
encoders are fakes."""

import http.client
import json
import threading
import time

import pytest

from bebop_vision import videoserver as vs
from bebop_vision.whep_registry import PeerWatch, ViewerLimit, WhepRegistry


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeHub:
    def __init__(self, name):
        self.name = name
        self.peers = set()
        self.closed = False
        self.fail = None

    def add_peer(self, p):
        self.peers.add(p)

    def remove_peer(self, p):
        self.peers.discard(p)

    def poll(self, now):
        return self.fail

    def close(self):
        self.closed = True


class FakePeer:
    def __init__(self):
        self.closed = False
        self.reason = None

    def poll(self, now):
        return self.reason

    def close(self):
        self.closed = True


def make_registry(max_viewers=4, grace=5.0):
    hubs = []

    def make_hub(name):
        h = FakeHub(name)
        hubs.append(h)
        return h

    clock = Clock()
    reg = WhepRegistry(make_hub, max_viewers=max_viewers,
                       release_grace_s=grace, clock=clock,
                       log=lambda *a: None)
    return reg, hubs, clock


def join(reg, stream="color_near"):
    sid = reg.reserve(stream)
    hub = reg.hub(stream)
    peer = FakePeer()
    assert reg.attach(sid, peer)
    return sid, peer, hub


def test_viewers_share_one_encoder():
    reg, hubs, _ = make_registry()
    joined = [join(reg) for _ in range(3)]
    assert len(hubs) == 1
    assert hubs[0].peers == {p for _, p, _ in joined}
    st = reg.stats()["color_near"]
    assert st["viewers"] == 3 and st["encoder"] is True
    assert len(st["sessions"]) == 3


def test_viewer_cap_per_stream():
    reg, hubs, _ = make_registry(max_viewers=2)
    join(reg)
    join(reg)
    with pytest.raises(ViewerLimit) as ei:
        reg.reserve("color_near")
    assert ei.value.viewers == 2 and ei.value.max_viewers == 2
    join(reg, "depth_near")            # other streams are independent
    assert len(hubs) == 2


def test_reservations_count_toward_cap():
    reg, _, _ = make_registry(max_viewers=1)
    sid = reg.reserve("color_near")    # still negotiating
    with pytest.raises(ViewerLimit):
        reg.reserve("color_near")
    reg.remove(sid, "negotiation failed")
    reg.reserve("color_near")


def test_delete_tears_down_peer_now_and_frees_slot():
    reg, hubs, _ = make_registry(max_viewers=1)
    sid, peer, hub = join(reg)
    assert reg.remove(sid)
    assert peer.closed and peer not in hub.peers
    assert not reg.remove(sid)         # second DELETE: unknown
    assert not reg.has(sid)
    join(reg)                          # slot is free again


def test_last_viewer_releases_encoder_after_grace():
    reg, hubs, clock = make_registry(grace=5.0)
    a, _, hub = join(reg)
    b, _, _ = join(reg)
    reg.remove(a)
    clock.t += 60
    reg.reap()
    assert not hub.closed              # b still watching
    reg.remove(b)
    clock.t += 4.9
    reg.reap()
    assert not hub.closed              # within grace
    clock.t += 0.2
    reg.reap()
    assert hub.closed
    assert reg.stats() == {}
    join(reg)                          # fresh encoder for the next viewer
    assert len(hubs) == 2


def test_viewer_returning_within_grace_reuses_encoder():
    reg, hubs, clock = make_registry(grace=5.0)
    sid, _, hub = join(reg)
    reg.remove(sid)
    clock.t += 3
    _, _, hub2 = join(reg)
    assert hub2 is hub
    clock.t += 10
    reg.reap()
    assert not hub.closed and len(hubs) == 1


def test_zero_grace_releases_immediately():
    reg, hubs, _ = make_registry(grace=0.0)
    sid, _, hub = join(reg)
    reg.remove(sid)
    assert hub.closed


def test_reap_drops_dead_peers_only():
    reg, _, _ = make_registry()
    a, pa, hub = join(reg)
    b, pb, _ = join(reg)
    pa.reason = "peer failed"
    reg.reap()
    assert pa.closed and not reg.has(a)
    assert not pb.closed and reg.has(b)
    assert hub.peers == {pb}


def test_failed_encoder_drops_all_viewers_and_is_replaced():
    reg, hubs, _ = make_registry()
    a, pa, hub = join(reg)
    b, pb, _ = join(reg)
    hub.fail = "pipeline error: boom"
    reg.reap()
    assert hub.closed and pa.closed and pb.closed
    assert reg.viewers("color_near") == 0
    _, _, hub2 = join(reg)
    assert hub2 is not hub


def test_attach_after_delete_closes_peer():
    reg, _, _ = make_registry()
    sid = reg.reserve("color_near")
    reg.hub("color_near")
    reg.remove(sid)                    # DELETE raced the negotiation
    peer = FakePeer()
    assert not reg.attach(sid, peer)
    assert peer.closed


def test_hub_creation_failure_propagates_and_keeps_no_hub():
    def boom(name):
        raise RuntimeError("no frame")

    reg = WhepRegistry(boom, release_grace_s=0.0, log=lambda *a: None)
    sid = reg.reserve("color_near")
    with pytest.raises(RuntimeError):
        reg.hub("color_near")
    reg.remove(sid)
    assert reg.stats() == {}


def test_close_all():
    reg, hubs, _ = make_registry()
    _, pa, hub = join(reg)
    _, pb, hubd = join(reg, "depth_far")
    reg.close_all()
    assert pa.closed and pb.closed and hub.closed and hubd.closed
    assert reg.stats() == {}


def test_concurrent_reserve_respects_cap():
    reg, _, _ = make_registry(max_viewers=4)
    got, refused = [], []

    def worker():
        try:
            got.append(reg.reserve("color_near"))
        except ViewerLimit:
            refused.append(1)

    ts = [threading.Thread(target=worker) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 4 and len(refused) == 12


# ------------------------------------------------------------- PeerWatch
def test_peerwatch_connect_timeout():
    w = PeerWatch(0.0, connect_timeout_s=15)
    assert w.verdict("new", 10) is None
    assert w.verdict("connecting", 15) == "connect timeout"


def test_peerwatch_failed_and_closed_are_immediate():
    assert PeerWatch(0.0).verdict("failed", 0.1) == "peer failed"
    assert PeerWatch(0.0).verdict("closed", 0.1) == "peer closed"


def test_peerwatch_disconnect_grace_and_recovery():
    w = PeerWatch(0.0, disconnect_timeout_s=5, idle_timeout_s=100)
    assert w.verdict("connected", 1) is None
    assert w.verdict("disconnected", 2) is None
    assert w.verdict("connected", 4) is None        # ICE recovered
    assert w.verdict("disconnected", 5) is None
    assert w.verdict("disconnected", 9.9) is None
    assert w.verdict("disconnected", 10) == "disconnect timeout"


def test_peerwatch_rtcp_silence_and_bye():
    w = PeerWatch(0.0, rtcp_timeout_s=8, idle_timeout_s=100)
    w.verdict("connected", 1)
    assert w.verdict("connected", 50) is None        # never saw RTCP: unarmed
    w.on_rtcp(50)
    assert w.verdict("connected", 57) is None
    assert w.verdict("connected", 58) == "rtcp timeout"
    w2 = PeerWatch(0.0)
    w2.on_bye()
    assert w2.verdict("connected", 1) == "rtcp bye"


def test_peerwatch_media_idle():
    w = PeerWatch(0.0, idle_timeout_s=8)
    w.verdict("connected", 1)
    assert w.verdict("connected", 8.9) is None
    assert w.verdict("connected", 9) == "idle timeout"
    w.on_media(9.5)
    assert w.verdict("connected", 10) is None


# ------------------------------------------------------------- _StreamHub
class FakeEncoder:
    def __init__(self, on_au=None, **kw):
        self.kw = kw
        self.on_au = on_au
        self.pushed = []
        self.keyframes = 0
        self.closed = False
        self.last_au_at = None

    def push(self, data):
        self.pushed.append(data)
        self.last_au_at = time.monotonic()
        self.on_au(b"au%d" % len(self.pushed), len(self.pushed) == 1)

    def request_keyframe(self):
        self.keyframes += 1
        return True

    def take_error(self):
        return None

    def close(self):
        self.closed = True


class HubPeer:
    def __init__(self):
        self.aus = []

    def push_au(self, au, idr):
        self.aus.append((au, idr))


class Frame:
    fps = 30.0
    depth = None

    def __init__(self, n):
        self.color_jpeg = b"jpg%d" % n


class Cam:
    def __init__(self):
        self.n = 0

    def read(self):
        self.n += 1
        return Frame(self.n)


def _wait(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_stream_hub_fans_out_and_idles_without_viewers():
    encs = []

    def make_encoder(**kw):
        e = FakeEncoder(**kw)
        encs.append(e)
        return e

    hub = vs._StreamHub("color_near", Cam(), "color", 3_500_000,
                        make_encoder=make_encoder)
    try:
        enc = encs[0]
        assert enc.kw["bitrate"] == 3_500_000
        assert enc.kw["input_kind"] == "jpeg"
        time.sleep(0.15)
        assert enc.pushed == []                 # no viewers: no encoding
        a, b = HubPeer(), HubPeer()
        hub.add_peer(a)
        hub.add_peer(b)
        assert _wait(lambda: len(a.aus) >= 3 and len(b.aus) >= 3)
        assert a.aus[0] == (b"au1", True)
        hub.keyframe()
        assert _wait(lambda: enc.keyframes == 1)
        hub.remove_peer(a)
        hub.remove_peer(b)
        time.sleep(0.1)
        n = len(enc.pushed)
        time.sleep(0.15)
        assert len(enc.pushed) == n             # paused again
        assert hub.poll(time.monotonic()) is None
    finally:
        hub.close()
    assert enc.closed


def test_stream_hub_without_frames_raises(monkeypatch):
    monkeypatch.setattr(vs, "FIRST_FRAME_TIMEOUT_S", 0.1)

    class Dead:
        def read(self):
            return None

    with pytest.raises(vs.NoFrame):
        vs._StreamHub("color_near", Dead(), "color", 1,
                      make_encoder=FakeEncoder)


# ------------------------------------------------------------ HTTP / WHEP
OFFER = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 102\r\na=rtpmap:102 H264/90000\r\n"


class HttpPeer(FakePeer):
    def __init__(self, on_keyframe_request=None, **kw):
        super().__init__()

    def stats(self):
        return {"state": "connected", "au_sent": 3}

    def negotiate(self, offer):
        assert "m=video" in offer
        return "v=0\r\nanswer\r\n"


@pytest.fixture
def whep_server(monkeypatch):
    import bebop_vision.whep as whep
    monkeypatch.setattr(whep, "webrtc_available", lambda: True)
    monkeypatch.setattr(whep, "WhepPeer", HttpPeer)

    class Rig:
        cameras = {}

    v = vs.VideoServer(Rig(), port=0, max_viewers=2, release_grace_s=5.0)
    hubs = []

    def make_hub(name):
        h = FakeHub(name)
        h.keyframe = lambda: None
        h.stats = lambda: {"frames": 9, "au_out": 9, "idr_out": 7,
                           "keyframe_requests": 1}
        hubs.append(h)
        return h

    v.registry._make_hub = make_hub
    v.start()
    yield v, v._httpd.server_address[1], hubs
    v.stop()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r, data


def _post_offer(port, stream="color_near", extra=""):
    return _req(port, "POST", f"/whep?stream={stream}{extra}", OFFER,
                {"Content-Type": "application/sdp"})


def test_whep_post_location_delete_and_cap(whep_server):
    v, port, hubs = whep_server
    r1, body = _post_offer(port)
    assert r1.status == 201
    assert r1.getheader("Content-Type") == "application/sdp"
    assert "Location" in r1.getheader("Access-Control-Expose-Headers")
    loc1 = r1.getheader("Location")
    assert loc1.startswith("/whep/") and body.startswith(b"v=0")
    r2, _ = _post_offer(port)
    assert r2.status == 201
    assert r2.getheader("Location") != loc1
    assert len(hubs) == 1 and len(hubs[0].peers) == 2

    r3, body = _post_offer(port)                 # cap = 2
    assert r3.status == 503
    assert r3.getheader("Retry-After") == "5"
    assert r3.getheader("Access-Control-Allow-Origin") == "*"
    err = json.loads(body)
    assert err["error"] == "viewer limit" and err["max_viewers"] == 2

    assert r3.getheader("Access-Control-Expose-Headers") \
        == "Location, Retry-After"

    r, body = _req(port, "GET", "/whep")
    cn = json.loads(body)["streams"]["color_near"]
    assert cn["viewers"] == 2 and cn["encoder"] is True
    assert cn["encoder_stats"]["idr_out"] == 7
    assert len(cn["sessions"]) == 2
    assert all(s["au_sent"] == 3 for s in cn["sessions"].values())

    r, _ = _req(port, "DELETE", loc1)
    assert r.status == 200
    r, _ = _req(port, "DELETE", loc1)
    assert r.status == 404
    r4, _ = _post_offer(port)                    # freed slot is reusable
    assert r4.status == 201


def test_whep_replace_frees_own_slot(whep_server):
    v, port, hubs = whep_server
    r1, _ = _post_offer(port)
    _post_offer(port)
    sid = r1.getheader("Location").rsplit("/", 1)[1]
    r, _ = _post_offer(port, extra=f"&replace={sid}")
    assert r.status == 201
    assert not v.registry.has(sid)


def test_whep_last_viewer_release(whep_server):
    v, port, hubs = whep_server
    r1, _ = _post_offer(port)
    _req(port, "DELETE", r1.getheader("Location"))
    assert not hubs[0].closed                    # grace
    v.registry.reap(now=time.monotonic() + 6)
    assert hubs[0].closed


def test_whep_cors_preflight_allows_delete(whep_server):
    _, port, _ = whep_server
    r, _ = _req(port, "OPTIONS", "/whep/abc")
    assert r.status == 204
    assert "DELETE" in r.getheader("Access-Control-Allow-Methods")
    r, _ = _req(port, "DELETE", "/whep/abc")
    assert r.status == 404                       # actual response exposes
    assert "Location" in r.getheader("Access-Control-Expose-Headers")


def test_whep_rejects_non_sdp_and_unavailable(whep_server, monkeypatch):
    _, port, _ = whep_server
    r, _ = _req(port, "POST", "/whep", "hello",
                {"Content-Type": "application/sdp"})
    assert r.status == 400
    import bebop_vision.whep as whep
    monkeypatch.setattr(whep, "webrtc_available", lambda: False)
    r, body = _post_offer(port)
    assert r.status == 503 and json.loads(body)["error"] == "webrtc unavailable"


def test_stale_reservation_is_reaped_and_frees_slot():
    reg, hubs, clock = make_registry(max_viewers=1)
    sid = reg.reserve("color_near")            # handler stalls negotiating
    with pytest.raises(ViewerLimit):
        reg.reserve("color_near")
    reg.reap(now=clock.t + 29)
    assert reg.has(sid)
    clock.t += 31
    reg.reap()
    assert not reg.has(sid)
    reg.reserve("color_near")                  # slot is free again
    late = FakePeer()                          # stalled handler finishes
    assert not reg.attach(sid, late) and late.closed


def test_hub_created_after_stream_released_is_closed():
    reg, hubs, clock = make_registry(grace=0.0)
    sid = reg.reserve("color_near")
    orig = reg._make_hub

    def slow_make(name):
        reg.remove(sid, "reaped mid-create")   # grace 0: stream dropped
        return orig(name)

    reg._make_hub = slow_make
    with pytest.raises(RuntimeError):
        reg.hub("color_near")
    assert hubs[0].closed
    with pytest.raises(RuntimeError):
        reg.hub("color_near")                  # no reservation at all


def test_stats_include_encoder_and_session_counters():
    reg, hubs, clock = make_registry()
    sid, peer, hub = join(reg)
    hub.stats = lambda: {"idr_out": 4, "keyframe_requests": 2}
    peer.stats = lambda: {"state": "connected", "au_sent": 10}
    pending = reg.reserve("color_near")
    st = reg.stats()["color_near"]
    assert st["viewers"] == 2 and st["encoder"] is True
    assert st["encoder_stats"] == {"idr_out": 4, "keyframe_requests": 2}
    assert st["sessions"][sid[:6]]["au_sent"] == 10
    assert st["sessions"][pending[:6]]["state"] == "negotiating"


def test_close_logs_peer_and_encoder_counters():
    logs = []
    reg = WhepRegistry(FakeHub, release_grace_s=0.0, clock=Clock(),
                       log=logs.append)
    sid, peer, hub = join(reg)
    hub.stats = lambda: {"frames": 5, "idr_out": 3, "keyframe_requests": 2,
                         "viewers": 1}
    peer.stats = lambda: {"state": "connected", "au_sent": 8}
    reg.remove(sid)
    closed = next(m for m in logs if "viewer" in m and "closed" in m)
    assert "au_sent=8" in closed and "idr_out=3" in closed
    assert "keyframe_requests=2" in closed and "viewers=1" not in closed
    released = next(m for m in logs if "encoder released" in m)
    assert "idr_out=3" in released
