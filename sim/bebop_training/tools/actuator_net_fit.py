#!/usr/bin/env python3
"""Fit learned Robstride torque-response models (hybrid actuator-net, td-b05f58).

Paper-derived (Dowdy & Chagas Vaz, arXiv:2607.18135): their zero-shot
sim-to-real transfer hinged on replacing the analytical actuator model with a
learned "actuator-net". Bebop's variable-impedance action makes the paper's
fixed-gain ``(pos_err, vel) -> torque`` net untrainable from existing data
(every sysid maneuver is torque-mode, kp=kd=0 — the gain dimension has no
coverage), so we decompose instead:

    desired  = kp*(q* - q) + kd*(0 - qd)        (analytic MIT PD, policy gains)
    realized = MLP(history(desired), history(qd))  <- THIS TOOL trains the MLP
    applied  = DC speed-torque envelope clip(realized)   (analytic rail, sim-side)

The net therefore only has to learn the drive-train response: torque constant,
current limiting / saturation, and driver lag as a function of commanded
torque and measured velocity. The sysid logs show this gap is LARGE at high
demand (RS04 cmd 36 Nm -> fb ~14 Nm; RS02 cmd 5.1 -> fb ~1.8) while the sim's
DCMotorCfg saturates at datasheet stall (120/60/17) — the sim overestimates
torque authority exactly where push recovery needs it.

Data
----
``sim/bebop-sysid-logs/sysid_<joint>_<maneuver>.csv`` produced by the
``sysid`` Rust binary (firmware/bebop-linux/src/bin/sysid.rs). Joints are
pooled per MOTOR MODEL (RS04: hip_flexion+knee, RS03: hip_abduction, RS02:
foot) so the left-side joints with no logs (hip_flexion_left) are covered by
the model-level net; per-episode friction/armature DR covers unit spread.

Method
------
Each maneuver is resampled to the sim physics rate (200 Hz) so the history
window means the same thing at train and sim time (the sim-side actuator
advances its history every ``compute()`` call, i.e. every physics step).
Features per sample: window of the last ``H`` samples of (cmd_tau/stall,
fb_vel/noload), newest first; target: fb_tau/stall. Architecture: 3x64 MLP
with softsign activations (the paper's net), Adam, MSE, early stopping on a
random 20% split; the ``torque-step`` maneuver is additionally REPORTED
per-model as a held-out-maneuver generalization metric.

For comparison we also score the analytic baseline = the DCMotor envelope
clip the sim uses today (same stall/effort-limit/noload constants), so the
report directly answers "how much better than the current model is the net".

Outputs (consumed by ``envs/bebop_v2_actuator_net.py``)
-------------------------------------------------------
``<out_dir>/<MODEL>.pt``   TorchScript net, input (batch, 2*H), output (batch, 1)
``<out_dir>/meta.json``    scales/window/metrics — must match the sim-side cfg

Runs inside the Isaac Lab container (needs torch), e.g.::

    /workspace/isaaclab/isaaclab.sh -p \\
        /workspace/bebop_bot/sim/bebop_training/tools/actuator_net_fit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Motor model constants — mirror exp_standing.py (datasheet values) and the
# effort_limit_sim rails. Kept local (like tools/sysid_fit.py) so this tool
# does not import the isaaclab-dependent training package.
# ---------------------------------------------------------------------------

# joint name -> motor model
JOINT_MAP = {
    "hip_flexion_left_joint": "RS04",
    "hip_flexion_right_joint": "RS04",
    "hip_abduction_left_joint": "RS03",
    "hip_abduction_right_joint": "RS03",
    "knee_flexion_left_joint": "RS04",
    "knee_flexion_right_joint": "RS04",
    "foot_left_joint": "RS02",
    "foot_right_joint": "RS02",
}

# Datasheet stall torque (Nm) / no-load speed (rad/s) — normalization scales.
STALL_TORQUE = {"RS04": 120.0, "RS03": 60.0, "RS02": 17.0}
NOLOAD_VEL = {"RS04": 20.9, "RS03": 20.4, "RS02": 42.9}

# exp_standing.py effort_limit_sim (the sim's applied-effort rail).
EFFORT_LIMIT_SIM = {"RS04": 84.0, "RS03": 42.0, "RS02": 17.0}

MODELS = ["RS04", "RS03", "RS02"]

# Maneuver held out of early-stopping weight updates? No — it IS trained on;
# we only REPORT its RMSE separately (leave-one-maneuver-out generalization
# would waste the widest-range maneuver; the random 20% split drives early
# stopping). Kept as a constant so tests can rely on it.
REPORT_MANEUVER = "torque_step"

HISTORY_LENGTH = 5  # samples @ sim rate -> 25 ms at 200 Hz
SIM_RATE_HZ = 200.0


# ---------------------------------------------------------------------------
# Data loading / feature building (numpy-only, unit-testable without torch)
# ---------------------------------------------------------------------------


@dataclass
class Maneuver:
    joint: str
    model: str
    kind: str  # e.g. "torque_chirp"
    t: np.ndarray
    cmd_tau: np.ndarray
    fb_vel: np.ndarray
    fb_tau: np.ndarray


def parse_log_filename(path: Path) -> tuple[str, str] | None:
    """``sysid_<joint>_<maneuver>.csv`` -> (joint, maneuver). None if not a log."""
    name = path.name
    if not (name.startswith("sysid_") and name.endswith(".csv")):
        return None
    stem = name[len("sysid_") : -len(".csv")]
    if "_joint_" not in stem:
        return None
    joint_part, maneuver = stem.split("_joint_", 1)
    return joint_part + "_joint", maneuver


def load_logs(logs_dir: Path) -> list[Maneuver]:
    """Load every sysid CSV, resampled to SIM_RATE_HZ."""
    maneuvers: list[Maneuver] = []
    for path in sorted(logs_dir.glob("sysid_*.csv")):
        parsed = parse_log_filename(path)
        if parsed is None:
            continue
        joint, kind = parsed
        model = JOINT_MAP.get(joint)
        if model is None:
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if len(rows) < HISTORY_LENGTH + 2:
            continue
        t = np.array([float(r["t_s"]) for r in rows])
        cmd = np.array([float(r["cmd_tau"]) for r in rows])
        vel = np.array([float(r["fb_vel"]) for r in rows])
        tau = np.array([float(r["fb_tau"]) for r in rows])
        # Resample to the sim rate so the history window duration matches
        # sim-side compute() cadence. Logs are ~125 Hz; sim physics is 200 Hz.
        dt = 1.0 / SIM_RATE_HZ
        t_uniform = np.arange(t[0], t[-1] - dt, dt)
        if len(t_uniform) < HISTORY_LENGTH + 2:
            continue
        maneuvers.append(
            Maneuver(
                joint=joint,
                model=model,
                kind=kind,
                t=t_uniform,
                cmd_tau=np.interp(t_uniform, t, cmd),
                fb_vel=np.interp(t_uniform, t, vel),
                fb_tau=np.interp(t_uniform, t, tau),
            )
        )
    return maneuvers


def build_samples(
    maneuvers: list[Maneuver], model: str, window: int = HISTORY_LENGTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed (X, y, maneuver_idx) for one motor model, normalized.

    X[:, 0:H]       = cmd_tau history / stall   (newest first)
    X[:, H:2H]      = fb_vel  history / noload  (newest first)
    y               = fb_tau / stall
    maneuver_idx[i] = index into the filtered maneuver list (for per-maneuver
                      RMSE reporting).
    """
    group = [m for m in maneuvers if m.model == model]
    if not group:
        raise ValueError(f"no maneuvers for model {model}")
    stall = STALL_TORQUE[model]
    noload = NOLOAD_VEL[model]
    xs, ys, mids = [], [], []
    for mid, m in enumerate(group):
        n = len(m.t)
        cmd_s = m.cmd_tau / stall
        vel_s = m.fb_vel / noload
        tau_s = m.fb_tau / stall
        for i in range(window - 1, n):
            # newest-first history, mirroring the sim-side roll(1) buffers
            cmd_hist = cmd_s[i - window + 1 : i + 1][::-1]
            vel_hist = vel_s[i - window + 1 : i + 1][::-1]
            xs.append(np.concatenate([cmd_hist, vel_hist]))
            ys.append(tau_s[i])
            mids.append(mid)
    if not xs:
        raise ValueError(f"no samples for model {model}")
    return np.stack(xs), np.array(ys), np.array(mids)


def analytic_baseline(cmd_tau: np.ndarray, fb_vel: np.ndarray, model: str) -> np.ndarray:
    """What the sim assumes today: DCMotor envelope clip of the command.

    Mirrors ``DCMotor._clip_effort`` (actuator_pd.py) with the current
    exp_standing.py constants (saturation = datasheet stall, effort_limit_sim
    rail, velocity_limit = no-load speed).
    """
    sat = STALL_TORQUE[model]
    eff_lim = EFFORT_LIMIT_SIM[model]
    vlim = NOLOAD_VEL[model]
    v_at_lim = vlim * (1.0 + eff_lim / sat)
    v = np.clip(fb_vel, -v_at_lim, v_at_lim)
    top = np.minimum(sat * (1.0 - v / vlim), eff_lim)
    bottom = np.maximum(sat * (-1.0 - v / vlim), -eff_lim)
    return np.clip(cmd_tau, bottom, top)


def baseline_rmse_nm(maneuvers: list[Maneuver], model: str) -> dict[str, float]:
    """Per-maneuver + overall RMSE (Nm) of the analytic baseline."""
    group = [m for m in maneuvers if m.model == model]
    out: dict[str, float] = {}
    all_pred, all_true = [], []
    for m in group:
        pred = analytic_baseline(m.cmd_tau, m.fb_vel, model)
        rmse = float(np.sqrt(np.mean((pred - m.fb_tau) ** 2)))
        out[f"{m.joint}:{m.kind}"] = rmse
        all_pred.append(pred)
        all_true.append(m.fb_tau)
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    out["ALL"] = float(np.sqrt(np.mean((pred - true) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Training (torch)
# ---------------------------------------------------------------------------


def train_model(
    model: str,
    maneuvers: list[Maneuver],
    out_dir: Path,
    window: int = HISTORY_LENGTH,
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 30,
    seed: int = 0,
) -> dict:
    """Train one motor-model response net, export TorchScript, return metrics."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, y, mids = build_samples(maneuvers, model, window)
    group = [m for m in maneuvers if m.model == model]
    stall = STALL_TORQUE[model]

    # Random 80/20 split for early stopping.
    perm = rng.permutation(len(X))
    n_val = max(1, int(0.2 * len(X)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32).unsqueeze(1)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32)
    yva = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    net = torch.nn.Sequential(
        torch.nn.Linear(2 * window, 64),
        torch.nn.Softsign(),
        torch.nn.Linear(64, 64),
        torch.nn.Softsign(),
        torch.nn.Linear(64, 64),
        torch.nn.Softsign(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        net.train()
        shuffle = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = shuffle[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(loss_fn(net(Xva), yva)))
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()

    # Metrics in physical units (Nm), overall + per maneuver + held-out-style report.
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred_s = net(Xt).squeeze(1).numpy()
    pred = pred_s * stall
    metrics: dict[str, float] = {}
    for mid, m in enumerate(group):
        mask = mids == mid
        true = m.fb_tau[window - 1 :]
        rmse = float(np.sqrt(np.mean((pred[mask] - true) ** 2)))
        metrics[f"{m.joint}:{m.kind}"] = rmse
    all_true = np.concatenate([m.fb_tau[window - 1 :] for m in group])
    metrics["ALL"] = float(np.sqrt(np.mean((pred - all_true) ** 2)))

    # Export TorchScript — the sim-side actuator loads this exact file.
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{model}.pt"
    scripted = torch.jit.script(net)
    scripted.save(str(pt_path))

    return {
        "model": model,
        "samples": int(len(X)),
        "epochs_run": epoch + 1,
        "net_rmse_nm": metrics,
        "file": str(pt_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_logs = Path(__file__).resolve().parents[2] / "bebop-sysid-logs"
    default_out = (
        Path(__file__).resolve().parents[1] / "assets" / "actuator_nets"
    )
    parser.add_argument("--logs-dir", type=Path, default=default_logs)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--window", type=int, default=HISTORY_LENGTH)
    args = parser.parse_args()

    maneuvers = load_logs(args.logs_dir)
    if not maneuvers:
        sys.exit(f"no sysid logs found in {args.logs_dir}")
    print(f"loaded {len(maneuvers)} maneuvers from {args.logs_dir}")

    meta = {
        "window": args.window,
        "sim_rate_hz": SIM_RATE_HZ,
        "input_layout": "[cmd_tau/stall (H newest-first), fb_vel/noload (H newest-first)]",
        "output": "fb_tau/stall (multiply by stall)",
        "models": {},
    }

    for model in MODELS:
        print(f"\n=== {model} ===")
        base = baseline_rmse_nm(maneuvers, model)
        print(f"  analytic baseline RMSE (ALL): {base['ALL']:.3f} Nm")
        result = train_model(
            model, maneuvers, args.out_dir, window=args.window, epochs=args.epochs
        )
        print(f"  samples={result['samples']}  epochs={result['epochs_run']}")
        print(f"  net RMSE (ALL):               {result['net_rmse_nm']['ALL']:.3f} Nm")
        step_keys = [k for k in result["net_rmse_nm"] if k.endswith(REPORT_MANEUVER)]
        for k in step_keys:
            print(
                f"  {REPORT_MANEUVER} report [{k}]: net {result['net_rmse_nm'][k]:.3f} Nm"
                f"  vs baseline {base.get(k, float('nan')):.3f} Nm"
            )
        print(f"  wrote {result['file']}")
        meta["models"][model] = {
            "stall_torque": STALL_TORQUE[model],
            "noload_vel": NOLOAD_VEL[model],
            "effort_limit_sim": EFFORT_LIMIT_SIM[model],
            "net_rmse_nm": result["net_rmse_nm"],
            "baseline_rmse_nm": base,
        }

    meta_path = args.out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
