""""Is a voice/agent turn in flight?" probe for the router tick.

The capture/place router shares the voice cascade's brain, so background ticks
must not fire while a spoken or typed turn is being served. The voice gateway
(`bebop-voice`, :9093) publishes its in-flight turns on ``GET /healthz`` as
``turn_active`` (bool) and ``turns_active`` (int).

Decision table (never raises, never blocks longer than ``timeout``):

* the gateway answers with a turn signal -> use it;
* the gateway is **not running** (connection refused) -> not busy: there is
  no turn to collide with;
* the gateway answers but carries **no** turn signal (an old build), times
  out, errors, or returns garbage -> **busy** (fail closed): it is up, so a
  turn may be running, and a skipped background tick is cheap whereas a tick
  that steals the brain mid-conversation is not. Logged once per kind.
"""
from __future__ import annotations

import errno
import json
import socket
import urllib.error
import urllib.request

DEFAULT_VOICE_URL = "http://127.0.0.1:9093/healthz"

#: Explicit in-flight-turn signals we understand. `turn_active` /
#: `turns_active` are what the gateway publishes; the rest are tolerated
#: aliases. Order is irrelevant.
BUSY_KEYS: tuple[str, ...] = (
    "turn_active",
    "turns_active",
    "in_flight",
    "in_flight_turn",
    "processing",
    "speaking",
    "utterance_active",
    "busy",
)


def _truthy(value) -> bool:
    """Interpret a status value without treating the string "false" as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, dict):
        return any(_truthy(value.get(key)) for key in
                   ("active", "busy", "in_flight", "count", "turns"))
    return False


def _refused(exc: BaseException) -> bool:
    """True when nothing is listening (the voice service is stopped)."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, ConnectionRefusedError):
        return True
    return isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED


class VoiceBusyProbe:
    """Callable ``() -> bool`` suitable for ``RouterService(voice_busy=...)``."""

    def __init__(
        self,
        url: str = DEFAULT_VOICE_URL,
        *,
        timeout: float = 0.5,
    ) -> None:
        self.url = str(url)
        self.timeout = max(0.05, float(timeout))
        self._logged: set[str] = set()
        self.last_error: str | None = None
        self.last_state: bool | None = None

    @property
    def describe(self) -> str:
        return (f"voice:{self.url} (turn_active; down=idle, "
                f"unknown=busy, timeout={self.timeout:.2f}s)")

    def _log_once(self, kind: str, message: str) -> None:
        if kind not in self._logged:
            self._logged.add(kind)
            print(f"[router] {message}", flush=True)

    def _result(self, busy: bool, error: str | None = None) -> bool:
        self.last_state = busy
        self.last_error = error
        return busy

    def __call__(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self.url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a tick
            error = f"{type(exc).__name__}: {exc}"
            if _refused(exc):
                self._log_once("down", f"voice service not running ({error}); "
                               "treating as idle")
                return self._result(False, error)
            if isinstance(exc, (socket.timeout, TimeoutError)) or (
                    isinstance(exc, urllib.error.URLError)
                    and isinstance(exc.reason, (socket.timeout, TimeoutError))):
                self._log_once("timeout", f"voice-busy probe timed out ({error}); "
                               "assuming busy")
            else:
                self._log_once("error", f"voice-busy probe failed ({error}); "
                               "assuming busy")
            return self._result(True, error)

        if not isinstance(payload, dict):
            self._log_once("garbage", "voice /healthz returned a non-object; "
                           "assuming busy")
            return self._result(True, "non-object payload")
        present = [key for key in BUSY_KEYS if key in payload]
        if not present:
            self._log_once("nosignal", "voice /healthz exposes no in-flight-turn "
                           "signal (old build?); assuming busy")
            return self._result(True, "no turn signal")
        return self._result(any(_truthy(payload.get(key)) for key in present))
