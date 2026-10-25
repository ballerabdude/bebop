import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="bebop lab quickstart")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # bebop lab: from a recorded drive to a model

    The loop this notebook walks: **pick a session, pick modalities, derive
    labels, run a baseline, look at it in Rerun, cache features for a backbone,
    train a head, score it against the baseline.** Every cell is a plain call
    into `bebop_vision.lab`; copy what you need into your own notebook.

    Run with `cd bebop-vision && .venv/bin/marimo edit notebooks/lab_quickstart.py`.
    """)
    return


@app.cell
def _():
    import os
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import numpy as np
    import polars as pl
    from bebop_vision.lab import InputSpec
    from bebop_vision.lab.catalog import Catalog
    from bebop_vision.lab.session import Session

    return Catalog, InputSpec, Session, pl


@app.cell
def _(Catalog, mo):
    catalog = Catalog()
    local = catalog.local()
    mo.vstack([mo.md("## Sessions on this machine"), mo.ui.table(local.to_dicts(), selection="single")])
    return catalog, local


@app.cell
def _(local, mo):
    session_name = mo.ui.dropdown(options=sorted(local["name"].to_list()),
                                  value=sorted(local["name"].to_list())[0] if len(local) else None,
                                  label="session")
    session_name
    return (session_name,)


@app.cell
def _(Session, catalog, session_name):
    s = Session(catalog.resolve(session_name.value))
    s.index()  # one pass over the MCAP the first time; cached afterwards
    s.summary()
    return (s,)


@app.cell
def _(mo, s):
    ticks = s.ticks(hz=10.0)
    mo.vstack([mo.md(f"## {len(ticks)} ticks at 10 Hz"), mo.ui.table(ticks.frame().head(50).to_dicts())])
    return (ticks,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Pick modalities

    A model declares what it reads with an `InputSpec`. The dataset stays raw:
    colour at the recorded resolution (BGR uint8), depth in mm, IMU windows,
    state records. Ablating a modality is editing this tuple.
    """)
    return


@app.cell
def _(InputSpec, mo, ticks):
    spec = InputSpec(("color_near", "depth_near", "imu_gyro_near", "odom", "cmd_vel"),
                     history_s={"imu_gyro_near": 1.0})
    sample = ticks.select(*spec.modalities, history_s=spec.history_s)[len(ticks) // 2]
    mo.md(f"""
    sample at t={sample.t_ns}: colour {None if sample['color_near'] is None else sample['color_near'].shape},
    depth {sample['depth_near'].shape}, gyro window {len(sample['imu_gyro_near'])} samples,
    odom {sample['odom']}, cmd {sample['cmd_vel']}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Labels from the recording itself

    Hindsight paths: where the robot actually went in the next 3 s, in the body
    frame of the tick, heading from the camera gyro (odom heading over-reads
    turns on carpet). Reverse windows are truncated and flagged.
    """)
    return


@app.cell
def _(s, ticks):
    from bebop_vision.lab import labels
    hind = labels.label_session(s, variant="hindsight", ticks=ticks)
    int(hind.valid.sum()), len(hind.t_ns)
    return (hind,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A baseline run, scored
    """)
    return


@app.cell
def _(hind, s, ticks):
    from bebop_vision.lab import metrics, models, runner
    cv = models.ConstantVelocity(horizon_s=3.0, dt_s=0.2)
    cv_run = runner.run(s, cv, ticks=ticks)
    metrics.score_paths(cv_run, hind)
    return cv_run, metrics, models, runner


@app.cell
def _(mo):
    mo.md(r"""
    ## Look at it

    Opens the native Rerun viewer with the recording, the labels and the run on
    the same timeline (needs the `rerun` viewer 0.31.3 on PATH).
    """)
    return


@app.cell
def _(mo):
    show_button = mo.ui.run_button(label="open in Rerun")
    show_button
    return (show_button,)


@app.cell
def _(cv_run, hind, s, show_button):
    if show_button.value:
        from bebop_vision.lab import rerun_log
        rerun_log.show(s, runs=[cv_run], labels=hind)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Architectures

    Any timm model, the DINOv3 ViTs through Hugging Face, or your own
    `nn.Module`, behind one interface. Frozen features are cached once per
    (backbone, geometry) so heads train in minutes.
    """)
    return


@app.cell
def _(pl):
    from bebop_vision.lab import backbones
    pl.DataFrame(backbones.available())
    return (backbones,)


@app.cell
def _(backbones, mo):
    backbone_name = mo.ui.dropdown(options=[b["name"] for b in backbones.available()],
                                   value="timm:convnext_tiny.dinov3_lvd1689m", label="backbone")
    pretrained = mo.ui.checkbox(value=False, label="pretrained weights (downloads once)")
    mo.hstack([backbone_name, pretrained])
    return backbone_name, pretrained


@app.cell
def _(backbone_name, backbones, pretrained, s, ticks):
    from bebop_vision.lab.features import FeatureCache
    bb = backbones.get(backbone_name.value, pretrained=pretrained.value)
    cache = FeatureCache(s.index())
    geometry = {"role": "far", "size": (416, 256), "crop": None}
    feat_dir = cache.compute(s, ticks, bb, geometry, what=("pooled",))
    feat_dir
    return (bb,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Train a head

    `HeadModel(backbone, head, mode)` with mode `frozen`, `lpft` (linear probe,
    then unfreeze the last blocks) or `full`. The run directory records the git
    hash, config, backbone id, label versions and split, so a leaderboard row
    is reproducible.
    """)
    return


@app.cell
def _(bb, hind, models, s, ticks):
    import torch
    from bebop_vision.lab import train
    from bebop_vision.lab.dataset import LabDataset
    # A head on the frozen far-camera features plus the commanded twist. One
    # drive is split by time here; across drives use dataset.make_splits.
    head = models.PathHead(in_dim=bb.embed_dim, n_scalars=2, n_points=15)
    path_model = models.HeadModel(bb, head, modality="color_far", size=(416, 256),
                                  scalar_modalities=("cmd_vel",), mode="frozen",
                                  name="convnext_path_v0")
    cut = int(len(ticks) * 0.7)
    tr = s.ticks(hz=10.0, start_ns=int(ticks.t_ns[0]), end_ns=int(ticks.t_ns[cut]))
    va = s.ticks(hz=10.0, start_ns=int(ticks.t_ns[cut + 1]), end_ns=int(ticks.t_ns[-1]))
    ds_tr = LabDataset([tr], path_model.spec, labels={"hindsight": hind})
    ds_va = LabDataset([va], path_model.spec, labels={"hindsight": hind})
    cfg = train.TrainConfig(epochs=3, batch=16, mode="frozen",
                            device="cuda" if torch.cuda.is_available() else "cpu")
    cfg
    return cfg, ds_tr, ds_va, path_model, train, va


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="train 3 epochs")
    train_button
    return (train_button,)


@app.cell
def _(cfg, ds_tr, ds_va, path_model, s, train, train_button):
    run_dir = None
    if train_button.value:
        run_dir = train.fit(path_model, ds_tr, ds_va, cfg, train.path_loss,
                            run_dir=s.index() / "train" / path_model.name)
        print(open(run_dir.path / "metrics.jsonl").read())
    run_dir
    return (run_dir,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Score it against the baseline on the held-out ticks

    Same labels, same ticks, one table. On one 67 s drive a frozen probe
    loses to constant velocity; that is the point of having the row.
    """)
    return


@app.cell
def _(hind, metrics, models, path_model, run_dir, runner, s, va):
    if run_dir is not None:
        rows = []
        for m in (path_model, models.ConstantVelocity()):
            r = runner.run(s, m, ticks=va)
            rows.append(({"model": m.name}, metrics.score_paths(r, hind)))
        board = metrics.leaderboard(rows)
    else:
        board = None
    board
    return


if __name__ == "__main__":
    app.run()
