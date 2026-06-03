#!/usr/bin/env python3
"""Fit Isaac Lab ``DCMotorCfg`` parameters from sysid CSV logs.

Companion to the ``sysid`` Rust binary in
``firmware/bebop-linux/src/bin/sysid.rs``. That binary drives a single
Robstride actuator through excitation maneuvers and logs synchronized CAN
feedback to CSV. This script reads those CSVs and fits the four parameters
that feed ``sim/bebop_training/experiments/exp_standing.py``:

================  =====================  ==========================
DCMotorCfg field  Maneuver               Fitted quantity
================  =====================  ==========================
``friction``      ``friction-sweep``     Coulomb friction torque (Nm)
``armature``      ``torque-step/chirp``  Reflected rotor inertia (kg.m^2)
``velocity_limit````noload-speed``       No-load terminal speed (rad/s)
``saturation_effort``  ``stall-torque``  Peak (stall) torque (Nm)
================  =====================  ==========================

``friction`` / ``armature`` are joint-level constants (per actuator group);
``saturation_effort`` / ``velocity_limit`` are motor-model-level constants.
Left/right joints (and both groups that share a motor model) are aggregated.

Usage
-----
    python sim/bebop_training/tools/sysid_fit.py ~/bebop-sysid-logs
    python sim/bebop_training/tools/sysid_fit.py run1.csv run2.csv

Only depends on numpy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Joint -> actuator group / motor model mapping (mirrors exp_standing.py).
# ---------------------------------------------------------------------------

# joint name -> (group suffix used in JOINT_* constants, Robstride model)
JOINT_MAP: dict[str, tuple[str, str]] = {
    "hip_flexion_left_joint": ("HIP_FLEX", "RS04"),
    "hip_flexion_right_joint": ("HIP_FLEX", "RS04"),
    "hip_abduction_left_joint": ("HIP_ABD", "RS03"),
    "hip_abduction_right_joint": ("HIP_ABD", "RS03"),
    "knee_flexion_left_joint": ("KNEE_FLEX", "RS04"),
    "knee_flexion_right_joint": ("KNEE_FLEX", "RS04"),
    "foot_left_joint": ("FOOT", "RS02"),
    "foot_right_joint": ("FOOT", "RS02"),
}

# Group suffix -> ordering for stable output.
GROUP_ORDER = ["HIP_FLEX", "HIP_ABD", "KNEE_FLEX", "FOOT"]
MODEL_ORDER = ["RS04", "RS03", "RS02"]

# Current values in exp_standing.py, shown for comparison / left unchanged
# when no data is available.
CURRENT_FRICTION = {"HIP_FLEX": 0.5, "HIP_ABD": 0.3, "KNEE_FLEX": 0.5, "FOOT": 0.1}
CURRENT_ARMATURE = {"HIP_FLEX": 0.025, "HIP_ABD": 0.012, "KNEE_FLEX": 0.025, "FOOT": 0.004}
CURRENT_STALL = {"RS04": 120.0, "RS03": 60.0, "RS02": 17.0}
CURRENT_NOLOAD = {"RS04": 20.9, "RS03": 20.4, "RS02": 42.9}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "t_s", "cmd_pos", "cmd_vel", "cmd_tau", "cmd_kp", "cmd_kd",
    "fb_pos", "fb_vel", "fb_tau", "fb_temp",
]


@dataclass
class Run:
    """One contiguous maneuver run for one joint (numpy arrays per column)."""
    joint: str
    maneuver: str
    model: str
    data: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data.get("t_s", ()))


def expand_inputs(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        elif any(ch in p for ch in "*?["):
            files.extend(sorted(glob.glob(p)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"warning: input {p!r} not found, skipping", file=sys.stderr)
    return files


def load_runs(files: list[str]) -> list[Run]:
    """Load CSVs, splitting into one Run per (file, joint, maneuver)."""
    runs: list[Run] = []
    for path in files:
        # group rows within the file by (joint, maneuver)
        cols: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        models: dict[tuple[str, str], str] = {}
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                missing = [c for c in ("joint", "maneuver") if c not in (reader.fieldnames or [])]
                if missing:
                    print(f"warning: {path}: missing columns {missing}, skipping", file=sys.stderr)
                    continue
                for row in reader:
                    key = (row["joint"], row["maneuver"])
                    models[key] = row.get("model", "")
                    for c in NUMERIC_COLS:
                        if c in row and row[c] != "":
                            try:
                                cols[key][c].append(float(row[c]))
                            except ValueError:
                                cols[key][c].append(math.nan)
        except OSError as e:
            print(f"warning: cannot read {path}: {e}", file=sys.stderr)
            continue

        for (joint, maneuver), colmap in cols.items():
            data = {c: np.asarray(v, dtype=float) for c, v in colmap.items()}
            if not data.get("t_s", np.array([])).size:
                continue
            runs.append(Run(joint=joint, maneuver=maneuver, model=models.get((joint, maneuver), ""), data=data))
    return runs


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if y.size < 2:
        return np.zeros_like(y)
    return np.gradient(y, t)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

@dataclass
class FrictionFit:
    coulomb: float
    viscous: float
    gravity: float
    n: int
    both_signs: bool


def fit_friction(run: Run) -> FrictionFit | None:
    """Fit tau_steady(v) = tau_c*sign(v) + b*v + g over moving, settled samples.

    The bidirectional sweep makes the constant gravity term ``g`` separable
    from Coulomb friction ``tau_c``.
    """
    t = run.data["t_s"]
    v = run.data["fb_vel"]
    tau = run.data["fb_tau"]
    if t.size < 10:
        return None

    # Steady-state, clearly-moving samples only.
    v_s = moving_average(v, 5)
    a = derivative(v_s, t)
    v_min = 0.15  # rad/s
    a_max = max(0.5, 0.1 * np.nanmax(np.abs(v_s)) + 1e-6)  # rad/s^2
    mask = (np.abs(v) > v_min) & (np.abs(a) < a_max) & np.isfinite(tau)
    if mask.sum() < 8:
        return None

    vm = v[mask]
    sign = np.sign(vm)
    X = np.column_stack([sign, vm, np.ones_like(vm)])
    coeffs, *_ = np.linalg.lstsq(X, tau[mask], rcond=None)
    tau_c, b, g = coeffs
    both = (sign > 0).any() and (sign < 0).any()
    return FrictionFit(coulomb=abs(float(tau_c)), viscous=float(b), gravity=float(g),
                       n=int(mask.sum()), both_signs=bool(both))


@dataclass
class ArmatureFit:
    inertia: float
    coulomb: float
    viscous: float
    gravity: float
    n: int


def fit_armature(run: Run) -> ArmatureFit | None:
    """Fit tau_applied = J*a + b*v + tau_c*sign(v) + g.

    Uses the open-loop torque maneuvers (torque-step / torque-chirp) where
    ``fb_tau`` is the applied torque and acceleration is excited. ``J`` is the
    reflected rotor inertia at the joint -> Isaac Lab ``armature``.
    """
    t = run.data["t_s"]
    v = run.data["fb_vel"]
    tau = run.data["fb_tau"]
    if t.size < 20:
        return None

    v_s = moving_average(v, 5)
    a = derivative(v_s, t)
    # Keep dynamically-rich samples where acceleration is observable.
    a_rms = math.sqrt(float(np.nanmean(a ** 2))) if a.size else 0.0
    a_min = max(0.2, 0.2 * a_rms)
    mask = (np.abs(a) > a_min) & np.isfinite(tau) & np.isfinite(v)
    if mask.sum() < 10:
        return None

    am = a[mask]
    vm = v[mask]
    X = np.column_stack([am, vm, np.sign(vm), np.ones_like(vm)])
    coeffs, *_ = np.linalg.lstsq(X, tau[mask], rcond=None)
    J, b, tau_c, g = coeffs
    if not np.isfinite(J) or J <= 0:
        return None
    return ArmatureFit(inertia=float(J), coulomb=abs(float(tau_c)), viscous=float(b),
                       gravity=float(g), n=int(mask.sum()))


def fit_noload_speed(run: Run) -> float | None:
    """Terminal no-load speed = high percentile of |velocity| (rad/s)."""
    v = run.data.get("fb_vel")
    if v is None or v.size < 5:
        return None
    speed = np.abs(v[np.isfinite(v)])
    if speed.size < 5:
        return None
    return float(np.percentile(speed, 98))


@dataclass
class StallFit:
    peak_torque: float
    max_speed: float  # to detect a shaft that wasn't actually blocked


def fit_stall_torque(run: Run) -> StallFit | None:
    """Peak (stall) torque = high percentile of |torque| (Nm)."""
    tau = run.data.get("fb_tau")
    v = run.data.get("fb_vel")
    if tau is None or tau.size < 5:
        return None
    mag = np.abs(tau[np.isfinite(tau)])
    if mag.size < 5:
        return None
    max_speed = float(np.nanmax(np.abs(v))) if v is not None and v.size else float("nan")
    return StallFit(peak_torque=float(np.percentile(mag, 99)), max_speed=max_speed)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class JointResult:
    joint: str
    group: str
    model: str
    friction: float | None = None
    viscous: float | None = None
    armature: float | None = None
    noload: float | None = None
    stall: float | None = None
    notes: list[str] = field(default_factory=list)


def fit_all(runs: list[Run]) -> dict[str, JointResult]:
    results: dict[str, JointResult] = {}

    def res_for(joint: str) -> JointResult | None:
        if joint not in JOINT_MAP:
            print(f"warning: joint {joint!r} not in known map, skipping", file=sys.stderr)
            return None
        if joint not in results:
            group, model = JOINT_MAP[joint]
            results[joint] = JointResult(joint=joint, group=group, model=model)
        return results[joint]

    for run in runs:
        r = res_for(run.joint)
        if r is None:
            continue
        man = run.maneuver
        if man == "friction-sweep":
            ff = fit_friction(run)
            if ff is None:
                r.notes.append("friction-sweep: too few settled samples")
            else:
                r.friction = ff.coulomb
                r.viscous = ff.viscous
                if not ff.both_signs:
                    r.notes.append("friction: one-directional sweep; gravity not separable")
        elif man in ("torque-step", "torque-chirp"):
            af = fit_armature(run)
            if af is None:
                r.notes.append(f"{man}: insufficient acceleration excitation for armature fit")
            else:
                # Prefer the run with more usable samples if multiple.
                r.armature = af.inertia
                if r.friction is None and af.coulomb > 0:
                    r.friction = af.coulomb
                    r.notes.append("friction taken from torque maneuver (no friction-sweep)")
        elif man == "noload-speed":
            nl = fit_noload_speed(run)
            if nl is not None:
                r.noload = nl
        elif man == "stall-torque":
            sf = fit_stall_torque(run)
            if sf is not None:
                r.stall = sf.peak_torque
                if math.isfinite(sf.max_speed) and sf.max_speed > 2.0:
                    r.notes.append(
                        f"stall: shaft moved up to {sf.max_speed:.1f} rad/s - "
                        "may not be fully blocked; peak torque underestimated"
                    )
        else:
            r.notes.append(f"unknown maneuver {man!r} ignored")

    return results


def _mean_spread(vals: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: dict[str, JointResult]) -> None:
    if not results:
        print("No usable runs found.")
        return

    print("\n" + "=" * 72)
    print("PER-JOINT FITS")
    print("=" * 72)
    hdr = f"{'joint':<26} {'friction':>9} {'viscous':>9} {'armature':>10} {'noload':>8} {'stall':>8}"
    print(hdr)
    print("-" * len(hdr))
    for joint in JOINT_MAP:
        if joint not in results:
            continue
        r = results[joint]
        def fmt(x, w, p):
            return f"{x:>{w}.{p}f}" if x is not None else f"{'-':>{w}}"
        print(
            f"{joint:<26} {fmt(r.friction,9,3)} {fmt(r.viscous,9,4)} "
            f"{fmt(r.armature,10,5)} {fmt(r.noload,8,2)} {fmt(r.stall,8,2)}"
        )
    # Notes
    notes_printed = False
    for joint in JOINT_MAP:
        r = results.get(joint)
        if r and r.notes:
            if not notes_printed:
                print("\nnotes:")
                notes_printed = True
            for n in r.notes:
                print(f"  - {joint}: {n}")

    # --- aggregate per group (friction, armature) ---
    group_friction: dict[str, list[float]] = defaultdict(list)
    group_armature: dict[str, list[float]] = defaultdict(list)
    model_stall: dict[str, list[float]] = defaultdict(list)
    model_noload: dict[str, list[float]] = defaultdict(list)
    for r in results.values():
        if r.friction is not None:
            group_friction[r.group].append(r.friction)
        if r.armature is not None:
            group_armature[r.group].append(r.armature)
        if r.stall is not None:
            model_stall[r.model].append(r.stall)
        if r.noload is not None:
            model_noload[r.model].append(r.noload)

    print("\n" + "=" * 72)
    print("AGGREGATED (per actuator group / motor model)")
    print("=" * 72)

    def show_group(label, agg, current):
        for key in (GROUP_ORDER if label in ("friction", "armature") else MODEL_ORDER):
            if key in agg:
                m, lo, hi = _mean_spread(agg[key])
                cur = current.get(key)
                delta = f"  (was {cur})" if cur is not None else ""
                spread = f"  [n={len(agg[key])}, range {lo:.4g}..{hi:.4g}]" if len(agg[key]) > 1 else ""
                print(f"  {label:<9} {key:<10} = {m:.5g}{delta}{spread}")

    show_group("friction", group_friction, CURRENT_FRICTION)
    show_group("armature", group_armature, CURRENT_ARMATURE)
    show_group("stall", model_stall, CURRENT_STALL)
    show_group("noload", model_noload, CURRENT_NOLOAD)

    # --- paste block ---
    print("\n" + "=" * 72)
    print("READY-TO-PASTE  (sim/bebop_training/experiments/exp_standing.py)")
    print("=" * 72)
    print("# Lines without measured data are emitted as comments (current value kept).")
    print()

    def emit(name_fmt, keys, agg, current, prec):
        for key in keys:
            const = name_fmt.format(key)
            if key in agg:
                m, _, _ = _mean_spread(agg[key])
                print(f"{const} = {round(m, prec)}")
            else:
                print(f"# {const} = {current.get(key)}  # no data - unchanged")

    print("# Coulomb friction (Nm), per actuator group")
    emit("JOINT_FRICTION_{}", GROUP_ORDER, group_friction, CURRENT_FRICTION, 3)
    print("\n# Reflected rotor inertia (kg.m^2), per actuator group")
    emit("JOINT_ARMATURE_{}", GROUP_ORDER, group_armature, CURRENT_ARMATURE, 5)
    print("\n# Stall torque (Nm), per motor model")
    emit("MOTOR_STALL_TORQUE_{}", MODEL_ORDER, model_stall, CURRENT_STALL, 2)
    print("\n# No-load speed (rad/s), per motor model")
    emit("MOTOR_NOLOAD_VEL_{}", MODEL_ORDER, model_noload, CURRENT_NOLOAD, 2)

    print("\nReminder: effort_limit_sim / velocity_limit_sim are deploy safety caps")
    print("(tied to bebop_v2.yaml hard_limits), NOT sysid targets - leave them unless")
    print("the envelope itself changes. Retrain after updating these constants.")
    print("Isaac Lab DCMotorCfg has no viscous-friction field; the 'viscous' column is")
    print("informational (it folds into the policy's learned kd / damping).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit DCMotorCfg params from sysid CSV logs.")
    ap.add_argument(
        "inputs",
        nargs="+",
        help="CSV files, globs, or directories (e.g. ~/bebop-sysid-logs).",
    )
    args = ap.parse_args(argv)

    files = expand_inputs(args.inputs)
    if not files:
        print("error: no CSV files found", file=sys.stderr)
        return 1
    print(f"Loaded {len(files)} CSV file(s):")
    for f in files:
        print(f"  {f}")

    runs = load_runs(files)
    if not runs:
        print("error: no usable runs parsed from CSVs", file=sys.stderr)
        return 1

    results = fit_all(runs)
    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
