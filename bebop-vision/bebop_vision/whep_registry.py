"""WHEP session bookkeeping: viewers per stream, shared encoders, lifecycle.

Pure Python (no GStreamer, no cameras) so the policy is unit-testable; the
media objects are injected:

  hub   one shared encoder per stream (videoserver._StreamHub). Interface:
        add_peer(peer), remove_peer(peer), poll(now) -> reason|None, close()
  peer  one WebRTC viewer (whep.WhepPeer). Interface:
        poll(now) -> reason|None, close()

Policy:
  * at most `max_viewers` viewers (reserved or live) per stream; the next
    POST gets `ViewerLimit` (answered as HTTP 503 + JSON);
  * the first viewer of a stream creates its hub; later viewers share it,
    so a viewer never adds an encoder;
  * the last viewer leaving arms a release timer; the hub is closed after
    `release_grace_s` unless another viewer arrives first (stream flips and
    reconnects reuse the warm encoder);
  * a reservation that never attaches a peer (a POST stuck negotiating)
    is dropped after `reserve_timeout_s` so it can't pin a slot;
  * `reap()` (called periodically) drops peers whose `poll()` reports a
    reason (ICE failure, disconnect timeout, RTCP silence, ...) and closes
    hubs that failed or whose grace expired.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

DEFAULT_MAX_VIEWERS = 4
DEFAULT_RELEASE_GRACE_S = 5.0
# A reservation whose POST never finished negotiating (a stalled handler)
# is dropped after this long so it can't hold a viewer slot forever.
DEFAULT_RESERVE_TIMEOUT_S = 30.0


class ViewerLimit(Exception):
    """The stream already has `max_viewers` viewers."""

    def __init__(self, stream, viewers, max_viewers):
        super().__init__(f"stream {stream} already has {viewers} viewers "
                         f"(max {max_viewers})")
        self.viewers = viewers
        self.max_viewers = max_viewers


@dataclass
class _Entry:
    stream: str
    created: float
    peer: object = None


@dataclass
class _Stream:
    sids: set = field(default_factory=set)
    hub: object = None
    release_at: float | None = None
    create_lock: threading.Lock = field(default_factory=threading.Lock)


class WhepRegistry:
    def __init__(self, make_hub, max_viewers=DEFAULT_MAX_VIEWERS,
                 release_grace_s=DEFAULT_RELEASE_GRACE_S,
                 reserve_timeout_s=DEFAULT_RESERVE_TIMEOUT_S,
                 clock=time.monotonic, log=print):
        self._make_hub = make_hub
        self.max_viewers = int(max_viewers)
        self.release_grace_s = float(release_grace_s)
        self.reserve_timeout_s = float(reserve_timeout_s)
        self._clock = clock
        self._log = log
        self._lock = threading.Lock()
        self._streams: dict[str, _Stream] = {}
        self._sessions: dict[str, _Entry] = {}

    # ---------------------------------------------------------- admission
    def reserve(self, stream) -> str:
        """Claim a viewer slot on `stream`; return the new session id.

        Raises ViewerLimit when full. The slot counts toward the cap
        while the offer is being negotiated; finish with attach() or
        remove()."""
        with self._lock:
            st = self._streams.setdefault(stream, _Stream())
            if len(st.sids) >= self.max_viewers:
                raise ViewerLimit(stream, len(st.sids), self.max_viewers)
            sid = secrets.token_urlsafe(12)
            while sid in self._sessions:
                sid = secrets.token_urlsafe(12)
            st.sids.add(sid)
            st.release_at = None          # a viewer is back: keep the hub
            self._sessions[sid] = _Entry(stream, self._clock())
            return sid

    def hub(self, stream):
        """The stream's shared hub, created on first use. Only call while
        holding a reservation on `stream` (that keeps the entry alive)."""
        with self._lock:
            st = self._streams.get(stream)
        if st is None:
            raise RuntimeError(f"stream {stream} has no reservation")
        with st.create_lock:            # serialize creation, not the registry
            with self._lock:
                if st.hub is not None:
                    return st.hub
            hub = self._make_hub(stream)
            with self._lock:
                orphaned = self._streams.get(stream) is not st
                if not orphaned:
                    st.hub = hub
            if orphaned:
                # Every reservation went away (reaped/deleted) while the
                # encoder was being built: don't leak it.
                _safe_close(hub)
                raise RuntimeError(f"stream {stream} released while "
                                   f"starting its encoder")
            self._log(f"[whep] {stream}: encoder started")
            return hub

    def attach(self, sid, peer) -> bool:
        """Bind a negotiated peer to its reservation and start feeding it.
        False (and the peer is closed) if the session was removed while
        negotiating."""
        with self._lock:
            entry = self._sessions.get(sid)
            st = self._streams.get(entry.stream) if entry else None
            hub = st.hub if st else None
            if entry is None or hub is None:
                ok = False
            else:
                entry.peer = peer
                ok = True
        if not ok:
            _safe_close(peer)
            return False
        hub.add_peer(peer)
        return True

    # ------------------------------------------------------------ teardown
    def remove(self, sid, reason="deleted") -> bool:
        """Tear down one viewer now. False if `sid` is unknown."""
        with self._lock:
            entry = self._sessions.pop(sid, None)
            if entry is None:
                return False
            st = self._streams.get(entry.stream)
            hub = st.hub if st else None
            if st is not None:
                st.sids.discard(sid)
                if not st.sids:
                    st.release_at = self._clock() + self.release_grace_s
            viewers = len(st.sids) if st else 0
        peer_stats = {}
        if entry.peer is not None:
            peer_stats = _safe_stats(entry.peer)
            if hub is not None:
                hub.remove_peer(entry.peer)
            _safe_close(entry.peer)
        else:
            peer_stats = {"state": "negotiating"}
        hub_stats = _safe_stats(hub) if hub is not None else {}
        self._log(f"[whep] {entry.stream}: viewer {sid[:6]} closed "
                  f"({reason}); {viewers} left; peer {_fmt(peer_stats)}; "
                  f"encoder {_fmt(hub_stats, _HUB_LOG_KEYS)}")
        if self.release_grace_s <= 0:
            self.reap()
        return True

    def reap(self, now=None):
        """Drop dead peers, failed hubs and hubs past their release grace."""
        now = self._clock() if now is None else now
        with self._lock:
            peers = [(sid, e.peer) for sid, e in self._sessions.items()
                     if e.peer is not None]
            stale = [sid for sid, e in self._sessions.items()
                     if e.peer is None
                     and now - e.created >= self.reserve_timeout_s]
            hubs = [(name, st.hub) for name, st in self._streams.items()
                    if st.hub is not None]
        for sid in stale:
            self.remove(sid, "negotiation timeout")
        for sid, peer in peers:
            reason = _safe_poll(peer, now)
            if reason:
                self.remove(sid, reason)
        for name, hub in hubs:
            reason = _safe_poll(hub, now)
            if reason:
                self._fail_stream(name, hub, reason)
        to_close = []
        with self._lock:
            for name, st in list(self._streams.items()):
                if st.sids or st.release_at is None or now < st.release_at:
                    continue
                if st.hub is not None:
                    to_close.append((name, st.hub))
                del self._streams[name]
        for name, hub in to_close:
            stats = _safe_stats(hub)
            _safe_close(hub)
            self._log(f"[whep] {name}: encoder released (no viewers); "
                      f"{_fmt(stats, _HUB_LOG_KEYS)}")

    def _fail_stream(self, name, hub, reason):
        """The shared encoder broke: drop every viewer so they reconnect
        onto a fresh hub, and close the broken one immediately."""
        with self._lock:
            st = self._streams.get(name)
            if st is None or st.hub is not hub:
                return
            st.hub = None
            sids = list(st.sids)
        self._log(f"[whep] {name}: encoder failed ({reason}); "
                  f"dropping {len(sids)} viewers; "
                  f"{_fmt(_safe_stats(hub), _HUB_LOG_KEYS)}")
        for sid in sids:
            self.remove(sid, f"encoder: {reason}")
        _safe_close(hub)

    def close_all(self):
        with self._lock:
            sids = list(self._sessions)
        for sid in sids:
            self.remove(sid, "server stopping")
        with self._lock:
            hubs = [st.hub for st in self._streams.values() if st.hub]
            self._streams.clear()
        for hub in hubs:
            _safe_close(hub)

    # --------------------------------------------------------------- views
    def has(self, sid) -> bool:
        with self._lock:
            return sid in self._sessions

    def viewers(self, stream) -> int:
        with self._lock:
            st = self._streams.get(stream)
            return len(st.sids) if st else 0

    def stats(self) -> dict:
        """Per stream: viewer count, the shared encoder's counters (None
        when no encoder runs) and each session's counters keyed by a short
        sid prefix (state "negotiating" while the offer is in flight)."""
        now = self._clock()
        with self._lock:
            snap = {name: (len(st.sids), st.hub,
                           [(sid, self._sessions[sid])
                            for sid in st.sids if sid in self._sessions])
                    for name, st in self._streams.items()}
        out = {}
        for name, (n, hub, entries) in snap.items():
            sessions = {}
            for sid, e in entries:
                s = (_safe_stats(e.peer) if e.peer is not None
                     else {"state": "negotiating"})
                s["age_s"] = round(now - e.created, 1)
                sessions[sid[:6]] = s
            out[name] = {"viewers": n, "encoder": hub is not None,
                         "encoder_stats": (_safe_stats(hub)
                                           if hub is not None else None),
                         "sessions": sessions}
        return out


_HUB_LOG_KEYS = ("frames", "au_out", "idr_out", "keyframe_requests")


def _safe_stats(obj) -> dict:
    try:
        fn = getattr(obj, "stats", None)
        return dict(fn()) if fn is not None else {}
    except Exception as exc:  # noqa: BLE001 - stats must not break teardown
        return {"stats_error": str(exc)}


def _fmt(stats, keys=None) -> str:
    items = [(k, stats[k]) for k in (keys or stats) if k in stats]
    return " ".join(f"{k}={v}" for k, v in items) or "-"


def _safe_close(obj):
    try:
        obj.close()
    except Exception as exc:  # noqa: BLE001 - teardown must not raise
        print(f"[whep] close failed: {exc}")


def _safe_poll(obj, now):
    try:
        return obj.poll(now)
    except Exception as exc:  # noqa: BLE001
        return f"poll error: {exc}"


class PeerWatch:
    """Liveness verdict for one viewer, from its connection state.

    `state` is the RTCPeerConnection state name ("new", "connecting",
    "connected", "disconnected", "failed", "closed"). A peer is dropped:
      * failed/closed: immediately;
      * never connected within `connect_timeout_s`;
      * disconnected for `disconnect_timeout_s` (ICE can recover from a
        short disconnect, e.g. a Wi-Fi roam);
      * RTCP silence for `rtcp_timeout_s` once RTCP was seen (browsers send
        receiver reports every ~1 s; a closed tab or a phone that lost the
        network goes silent long before ICE notices);
      * an RTCP BYE (the browser closed the connection);
      * no media delivered for `idle_timeout_s` while connected.
    """

    def __init__(self, now, connect_timeout_s=15.0, disconnect_timeout_s=5.0,
                 rtcp_timeout_s=8.0, idle_timeout_s=8.0):
        self.created = now
        self.connect_timeout_s = connect_timeout_s
        self.disconnect_timeout_s = disconnect_timeout_s
        self.rtcp_timeout_s = rtcp_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.connected_at = None
        self.disconnected_since = None
        self.last_rtcp = None
        self.last_media = None
        self.bye = False

    def on_rtcp(self, now):
        self.last_rtcp = now

    def on_bye(self):
        self.bye = True

    def on_media(self, now):
        self.last_media = now

    def verdict(self, state, now):
        if self.bye:
            return "rtcp bye"
        if state in ("failed", "closed"):
            return f"peer {state}"
        if state == "connected":
            if self.connected_at is None:
                self.connected_at = now
            self.disconnected_since = None
        elif state == "disconnected":
            if self.disconnected_since is None:
                self.disconnected_since = now
            if now - self.disconnected_since >= self.disconnect_timeout_s:
                return "disconnect timeout"
        if self.connected_at is None:
            if now - self.created >= self.connect_timeout_s:
                return "connect timeout"
            return None
        if (self.last_rtcp is not None
                and now - self.last_rtcp >= self.rtcp_timeout_s):
            return "rtcp timeout"
        last = self.last_media if self.last_media is not None \
            else self.connected_at
        if now - last >= self.idle_timeout_s:
            return "idle timeout"
        return None
