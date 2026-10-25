"""bebop lab: model development on recorded drives.

A Python package, not a UI. It turns the robot's drive MCAPs into typed,
time-aligned sessions, lets a model declare the modalities it wants, runs
any architecture (timm, Hugging Face or custom) over those samples in any
training mode (frozen, linear-probe-then-fine-tune, full, distillation),
derives labels from the recordings themselves, and shows everything on the
recording's own timeline in Rerun. Evaluation falls out of the same pieces.

Layout::

    types.py      shared dataclasses and protocols (Sample, InputSpec, Model, ...)
    session.py    Session / Ticks: one MCAP as lazy, aligned numpy series
    frames.py     H.265 / JPEG / PNG random access and sequential decode
    catalog.py    sessions on disk, on the robot (/captures) and in the ledger
    labels.py     labels derived from the recording (hindsight paths, ...)
    backbones.py  architecture registry (timm, HF, custom) behind one interface
    features.py   per-session frozen-feature cache keyed by backbone + geometry
    dataset.py    torch Dataset over (session, tick) pairs honouring an InputSpec
    models.py     Model protocol implementations: baselines, backbone + head
    train.py      training helper with run manifests
    runner.py     run a Model over a session; outputs saved per tick
    metrics.py    ADE/FDE, IoU, agreement; leaderboard
    rerun_log.py  show sessions, labels and runs in the bebop_navd Rerun app
    bench.py      ONNX export, parity, on-Thor trtexec latency
    cli.py        python -m bebop_vision.lab ...

Everything here reads the raw MCAP the recorder wrote; there is no second
dataset layout. Derived artifacts (indexes, caches, labels, features, runs)
live under a per-session cache directory and are keyed by the MCAP's
identity and the producer's version, so a stale artifact is detected, not
trusted.
"""

from bebop_vision.lab.types import (  # noqa: F401
    MODALITIES,
    Calib,
    CameraModel,
    Classes,
    Embedding,
    Grid,
    InputSpec,
    Mask,
    Model,
    Output,
    Path2D,
    Sample,
    Scalars,
    Series,
)

__all__ = [
    "MODALITIES", "Calib", "CameraModel", "Classes", "Embedding", "Grid",
    "InputSpec", "Mask", "Model", "Output", "Path2D", "Sample", "Scalars",
    "Series",
]
