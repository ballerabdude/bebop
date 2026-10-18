"""Fail-open "is a voice turn in flight?" probe for the router tick.

The capture/place router shares the voice cascade's brain, so background ticks
must not fire while a spoken turn is being served.  The voice gateway
(`bebop-voice`, :9093) currently exposes only build/readiness state on
``GET /healthz`` — it does not publish an in-flight-turn counter.  Rather than
guess from ``phase`` (a ready service is idle most of the time), this probe:

* looks for an explicit signal (``turn_active``, ``in_flight``, ``busy``,
  ``processing``, …) and uses it when present;
* otherwise **fails open**: assumes no turn is in flight and logs that fact
  once;
* on any transport/protocol error also fails open, logs once, and returns
  ``False`` — it can never raise or block the router thread beyond
  ``timeout`` seconds.

When the voice gateway grows a real turn counter, adding its key to
``BUSY_KEYS`` is the only change needed.
"""
from __future__ import annotations

import json
import urllib.request

DEFAULT_VOICE_URL = "http://127.0.0.1:9093/healthz"

#: Explicit in-flight-turn signals we understand.  Order is irrelevant.
BUSY_KEYS: tuple[str, ...] = (
    "turn_active",
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
        self._no_signal_logged = False
        self._error_logged = False
        self.last_error: str | None = None
        self.last_state: bool | None = None

    @property
    def describe(self) -> str:
        return f"voice:{self.url} (fail-open, timeout={self.timeout:.2f}s)"

    def __call__(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self.url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a tick
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_state = False
            if not self._error_logged:
                print(
                    f"[router] voice-busy probe failed ({self.last_error}); "
                    "assuming not busy (fail-open)",
                    flush=True,
                )
                self._error_logged = True
            return False

        if not isinstance(payload, dict):
            payload = {}
        present = [key for key in BUSY_KEYS if key in payload]
        if not present:
            self.last_error = None
            self.last_state = False
            if not self._no_signal_logged:
                print(
                    "[router] voice /healthz exposes no in-flight-turn signal; "
                    "assuming not busy (fail-open)",
                    flush=True,
                )
                self._no_signal_logged = True
            return False

        busy = any(_truthy(payload.get(key)) for key in present)
        self.last_error = None
        self.last_state = busy
        return busy
