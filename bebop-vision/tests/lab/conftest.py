"""Fixtures for the lab tests: synthetic sessions with known trajectories.

``synth.py`` beside this file writes them; import it as ``synth``.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # bebop-vision/ (bebop_vision, tools)
sys.path.insert(0, str(HERE))              # tests/lab/ (synth)

import synth  # noqa: E402


def _has_encoder(name):
    try:
        import av
        av.codec.Codec(name, "w")
        return True
    except Exception:
        return False


HAS_X265 = _has_encoder("libx265")
HAS_X264 = _has_encoder("libx264")

SESSION_KW = dict(duration_s=4.0, hz=20.0, v=0.3, w=0.4)


@pytest.fixture(scope="session")
def h265_session(tmp_path_factory):
    """A 4 s / 20 Hz H.265 session (the recorder's normal output) with
    runtime telemetry and notes."""
    if not HAS_X265:
        pytest.skip("PyAV has no libx265 encoder")
    d = tmp_path_factory.mktemp("h265")
    return synth.write_synthetic_session(d / "navd_session_20260101_000000.mcap",
                                         codec="h265", **SESSION_KW)


@pytest.fixture(scope="session")
def jpeg_session(tmp_path_factory):
    """The no-encoder path: CompressedImage JPEG colour, no runtime, no notes
    (like the 2026-10-18 recordings, which predate both)."""
    d = tmp_path_factory.mktemp("jpeg")
    return synth.write_synthetic_session(d / "navd_session_20260102_000000.mcap",
                                         codec=None, with_runtime=False,
                                         with_notes=False, **SESSION_KW)


@pytest.fixture(scope="session")
def dropped_far_session(tmp_path_factory):
    """H.265 with every 5th far colour frame missing."""
    if not HAS_X265:
        pytest.skip("PyAV has no libx265 encoder")
    d = tmp_path_factory.mktemp("drop")
    return synth.write_synthetic_session(d / "navd_session_20260103_000000.mcap",
                                         codec="h265", drop_far_every=5, **SESSION_KW)


@pytest.fixture(scope="session")
def reverse_session(tmp_path_factory):
    """JPEG session that reverses after 2.5 s (for label truncation tests)."""
    d = tmp_path_factory.mktemp("rev")
    return synth.write_synthetic_session(d / "navd_session_20260104_000000.mcap",
                                         codec=None, reverse_from_s=2.5, **SESSION_KW)


@pytest.fixture(scope="session")
def footerless_session(tmp_path_factory):
    """JPEG session cut before the footer (a power-cut capture)."""
    d = tmp_path_factory.mktemp("cut")
    return synth.write_synthetic_session(d / "navd_session_20260105_000000.mcap",
                                         codec=None, footer=False, **SESSION_KW)


@pytest.fixture
def lab_cache(tmp_path):
    """A fresh per-test lab cache root."""
    d = tmp_path / "lab-cache"
    d.mkdir()
    return d
