#!/usr/bin/env python3
"""Evaluate Bebop standing training runs from their TensorBoard scalar logs.

Codifies the convergence check used to judge a standing run before deploying:

  1. Trajectory of the three load-bearing scalars over training:
       - Train/mean_episode_length  (survival; the primary signal)
       - Train/mean_reward
       - Policy/mean_std            (exploration / std collapse)
  2. Summary stats: final value, global max (+ step), last-window mean, and the
     last-window slope (still improving vs plateaued vs degrading).
  3. Peak-vs-degradation detector: did mean_episode_length peak EARLY and then
     fall? (the reward-hacking / over-damped-freeze pattern we keep hitting).
  4. Per-term Episode_Reward breakdown at the last step, sorted by magnitude,
     so you can see which shaping term dominates the budget.
  5. A blunt verdict + the most likely issue.

Why these heuristics (from the standing-policy debugging history):
  - "Converged" is NOT "Policy/mean_std went flat". std collapsing while
    mean_episode_length is low/declining is PREMATURE COLLAPSE, not success.
  - The real convergence signal is mean_episode_length climbing to and HOLDING
    near its max (episode_length_s * control_hz, default 20 s * 100 Hz = 2000).
  - If eplen peaks early then degrades while a positive shaping term (e.g.
    stationary_pose) dominates, the policy is optimizing the shaping reward at
    the expense of survival.

Usage:
    pyenv shell 3.11.14   # an env with tensorboard installed
    # single run:
    python sim/tools/analyze_training.py sim/logs/rsl_rl/Isaac-BebopV2-Standing-v0/<run>
    # compare several runs side by side:
    python sim/tools/analyze_training.py <run_a> <run_b> <run_c>
    # options:
    python sim/tools/analyze_training.py <run> --max-eplen 2000 --last-frac 0.25
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Tag-name fallbacks (rsl_rl / Isaac Lab have shuffled these across versions).
EPLEN_TAGS = ["Train/mean_episode_length"]
REWARD_TAGS = ["Train/mean_reward"]
STD_TAGS = ["Policy/mean_std", "Policy/mean_noise_std"]


def _load(run_dir: str) -> EventAccumulator:
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _series(ea: EventAccumulator, tag: str):
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev], dtype=float)


def _pick(ea: EventAccumulator, candidates: list[str]):
    tags = ea.Tags()["scalars"]
    for t in candidates:
        if t in tags:
            return t
    return None


def _slope_per_1k(steps: np.ndarray, vals: np.ndarray, frac: float) -> float:
    """Linear slope (units per 1000 steps) over the last ``frac`` of the run."""
    if len(vals) < 3:
        return float("nan")
    k = max(2, int(len(vals) * frac))
    s, v = steps[-k:], vals[-k:]
    return float(np.polyfit(s, v, 1)[0] * 1000.0)


def _fmt_trajectory(steps, eplen, reward, std_steps, std_vals):
    targets = [1000, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 30000]
    last = steps[-1]
    targets = [t for t in targets if t <= last] + [last]
    seen = set()
    rows = []
    for t in targets:
        i = int(np.argmin(np.abs(steps - t)))
        if steps[i] in seen:
            continue
        seen.add(steps[i])
        j = int(np.argmin(np.abs(std_steps - steps[i]))) if len(std_steps) else 0
        ms = std_vals[j] if len(std_vals) else float("nan")
        rows.append((steps[i], eplen[i], reward[i], ms))
    return rows


def analyze_one(run_dir: str, max_eplen: float, last_frac: float):
    name = os.path.basename(os.path.normpath(run_dir))
    print("=" * 78)
    print(f"RUN: {name}")
    print(f"  {run_dir}")
    print("=" * 78)

    ea = _load(run_dir)
    eplen_tag = _pick(ea, EPLEN_TAGS)
    reward_tag = _pick(ea, REWARD_TAGS)
    std_tag = _pick(ea, STD_TAGS)
    if eplen_tag is None or reward_tag is None:
        print("  !! missing Train/mean_episode_length or Train/mean_reward — "
              "is this an rsl_rl run dir?")
        return

    steps, eplen = _series(ea, eplen_tag)
    _, reward = _series(ea, reward_tag)
    std_steps, std_vals = _series(ea, std_tag) if std_tag else (np.array([]), np.array([]))

    last = steps[-1]
    win = steps >= (last - last_frac * last)

    def summ(tag, s, v):
        i_max = int(np.argmax(v))
        return (f"  {tag:26s} final={v[-1]:8.2f}  max={v.max():8.2f}(@{s[i_max]:>6d})"
                f"  last{int(last_frac*100)}%_mean={v[win].mean():8.2f}"
                f"  slope/1k={_slope_per_1k(s, v, last_frac):+7.2f}")

    print(f"\n[iters] last_step={last}")
    print("\n[1] LOAD-BEARING SCALARS")
    print(summ("mean_episode_length", steps, eplen))
    print(summ("mean_reward", steps, reward))
    if len(std_vals):
        print(summ(std_tag.split("/")[-1], std_steps, std_vals))

    print("\n[2] TRAJECTORY")
    print(f"  {'step':>6} | {'eplen':>6} | {'reward':>7} | {'mean_std':>8}"
          f"   (eplen max possible = {max_eplen:.0f})")
    for s, e, r, m in _fmt_trajectory(steps, eplen, reward, std_steps, std_vals):
        print(f"  {s:6d} | {e:6.0f} | {r:7.2f} | {m:8.3f}")

    # [3] convergence / degradation logic
    print("\n[3] VERDICT")
    eplen_peak_i = int(np.argmax(eplen))
    eplen_peak_step = steps[eplen_peak_i]
    eplen_last = eplen[win].mean()
    eplen_max = eplen.max()
    std_final = std_vals[-1] if len(std_vals) else float("nan")
    surv_frac = eplen_last / max_eplen if max_eplen else float("nan")

    # heuristic thresholds
    held = surv_frac >= 0.97
    near = 0.90 <= surv_frac < 0.97
    early_peak = eplen_peak_step < 0.5 * last
    degraded = (eplen_max - eplen_last) / max(eplen_max, 1.0) > 0.12
    std_collapsed = (not np.isnan(std_final)) and std_final < 0.08

    if held:
        print(f"  + SOLVED: eplen holds {eplen_last:.0f}/{max_eplen:.0f} "
              f"({surv_frac*100:.0f}%) over the last {int(last_frac*100)}%.")
    elif near:
        print(f"  ~ MARGINAL: eplen ~{eplen_last:.0f}/{max_eplen:.0f} "
              f"({surv_frac*100:.0f}%) — stands most of the episode but still topples.")
    else:
        print(f"  - NOT SOLVED: eplen ~{eplen_last:.0f}/{max_eplen:.0f} "
              f"({surv_frac*100:.0f}%) — falls well before timeout.")

    if early_peak and degraded:
        print(f"  - DEGRADING: eplen peaked {eplen_max:.0f} @ {eplen_peak_step} "
              f"(first half) then fell to ~{eplen_last:.0f}. Reward-hacking / "
              f"over-damped freeze after std collapse — best checkpoint is EARLY.")
    if std_collapsed and not held:
        print(f"  ! std collapsed to {std_final:.3f} while eplen is "
              f"{eplen_last:.0f}/{max_eplen:.0f}: this is PREMATURE COLLAPSE, "
              f"not convergence (flat mean_std != solved).")
    if not std_collapsed and len(std_vals):
        print(f"  . mean_std={std_final:.3f} (not fully collapsed; still exploring).")

    # [4] reward-term breakdown
    rterms = sorted(t for t in ea.Tags()["scalars"] if t.startswith("Episode_Reward/"))
    if rterms:
        print("\n[4] EPISODE_REWARD TERMS @ last step (sorted by |value|)")
        vals = []
        for t in rterms:
            s, v = _series(ea, t)
            vals.append((t.split("/")[-1], v[-1]))
        vals.sort(key=lambda kv: abs(kv[1]), reverse=True)
        pos = [(k, v) for k, v in vals if v > 0]
        for k, v in vals:
            bar = "+" if v > 0 else "-"
            print(f"  {k:22s} {v:+.4f} {bar}")
        # flag a dominant positive shaping term (the freeze trap)
        shaping_pos = [(k, v) for k, v in pos if k not in ("alive",)]
        if shaping_pos:
            top_k, top_v = max(shaping_pos, key=lambda kv: kv[1])
            alive_v = dict(vals).get("alive", 0.0)
            if top_v > 0.4 * abs(alive_v) and not held:
                print(f"  ! '{top_k}' (+{top_v:.3f}) is a large positive shaping "
                      f"term vs alive (+{alive_v:.3f}); if survival is low this "
                      f"term may be steering the policy away from active balance.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+",
                    help="One or more rsl_rl run directories (contain events.out.tfevents*).")
    ap.add_argument("--max-eplen", type=float, default=2000.0,
                    help="Max possible mean_episode_length (episode_length_s * control_hz). "
                         "Default 2000 (20 s * 100 Hz).")
    ap.add_argument("--last-frac", type=float, default=0.25,
                    help="Trailing fraction of the run used for plateau/slope stats (default 0.25).")
    args = ap.parse_args()

    for rd in args.run_dirs:
        analyze_one(rd, args.max_eplen, args.last_frac)

    if len(args.run_dirs) > 1:
        _compare(args.run_dirs, args.max_eplen, args.last_frac)


def _compare(run_dirs, max_eplen, last_frac):
    print("=" * 78)
    print("COMPARISON (last-window means)")
    print("=" * 78)
    print(f"  {'run':32s} {'eplen':>8} {'%':>5} {'reward':>8} {'mean_std':>9}")
    for rd in run_dirs:
        try:
            ea = _load(rd)
            et = _pick(ea, EPLEN_TAGS); rt = _pick(ea, REWARD_TAGS); stt = _pick(ea, STD_TAGS)
            s, e = _series(ea, et)
            _, r = _series(ea, rt)
            win = s >= (s[-1] - last_frac * s[-1])
            ms = _series(ea, stt)[1][-1] if stt else float("nan")
            name = os.path.basename(os.path.normpath(rd))[:32]
            print(f"  {name:32s} {e[win].mean():8.0f} {e[win].mean()/max_eplen*100:5.0f}"
                  f" {r[win].mean():8.2f} {ms:9.3f}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {rd}: !! {ex}")
    print()


if __name__ == "__main__":
    main()
