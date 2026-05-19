# sim/

Simulation, RL training, and 3D assets for Bebop. Everything in here runs
**off-robot** on a workstation with an NVIDIA GPU and Isaac Sim / Isaac Lab.

## Layout

| Path               | What it is                                                                |
|--------------------|---------------------------------------------------------------------------|
| `pyproject.toml`   | Defines the `bebop_training` Python package (depends on `isaaclab`).      |
| `bebop_training/`  | RL extension: agents, envs, experiments, model export.                    |
| `usd/`             | USD scene assets (`bebopv2/` robot description for Isaac Sim).            |
| `scripts/`         | One-shot Isaac Sim utility scripts (run from the Script Editor).          |
| `docker/`          | Isaac Sim / Isaac Lab container image (single Dockerfile, two BASE_IMAGEs). |
| `logs/`            | Training logs (gitignored). Bind-mounted into the Isaac Lab container.    |

## Containers

The Isaac Sim and Isaac Lab containers are defined in the repo-root
`docker-compose.yml` and built from `sim/docker/Dockerfile` (the same image
recipe is reused with two different `BASE_IMAGE` args).

When the container starts, `sim/docker/entrypoint.sh` pip-installs the
`bebop_training` package in editable mode from `/workspace/bebop_bot/sim`,
which is the bind-mounted host path of this folder.

## Common commands

From the repo root:

```sh
just sim-up      # Isaac Sim + ROS 2 dev container (profile: sim)
just lab-up      # Isaac Lab + ROS 2 dev container (profile: lab)
just sim-down    # tear down sim profile
just lab-down    # tear down lab profile
```

Or directly with compose:

```sh
docker compose --profile sim up --build
docker compose --profile lab up --build
```

## URDF → USD import

After regenerating `ros2/src/bebopv2_description/urdf/bebopv2.urdf` (see
[`ros2/README.md`](../ros2/README.md) → "URDF mesh paths"), convert it
to USD inside the Isaac Lab container — Isaac Sim is launchable from
there via `/workspace/isaaclab/isaaclab.sh -s`. After the importer
finishes, run [`scripts/post_import_bebopv2.py`](scripts/post_import_bebopv2.py)
in **Window → Script Editor** to:

- disable the fixed root joint the importer adds (we need a
  free-floating base for the biped),
- ensure `base_link` is a dynamic (not kinematic) rigid body,
- translate the robot up by 0.65 m so it spawns standing on the ground
  plane (the URDF's `base_link` frame is at the hip, so without this
  the lower half of the robot is below `z=0`),
- attach a 200 Hz IMU prim (`Imu_Sensor`) under
  `<robot>/Geometry/base_link`, matching the on-robot BNO085 placement
  so the policy sees the same `/imu/data` shape in sim and on hardware.

The script is idempotent — safe to re-run.

## Sim-to-real contract (READ BEFORE TRAINING)

The Bebop V2 sim is pinned to the firmware's actual control behaviour, not
to whatever the motors *could* do mechanically. If you change either side
without changing the other, the trained policy will diverge on hardware.
Recent symptom of a divergence: "robot stands in sim, falls over instantly
on the real robot — feels weightless in sim".

The contract lives in three places that must agree:

| Quantity | Sim source of truth | Firmware source of truth |
|---|---|---|
| Per-joint policy kp/kd clamps | `POLICY_KP_MIN/MAX` and `POLICY_KD_MIN/MAX` in [`bebop_v2_base_cfg.py`](bebop_training/envs/bebop_v2_base_cfg.py) | `joints.<joint>.policy_gain_clamps` (per joint) + `defaults.policy_gain_clamps` in [`bebop_v2.yaml`](../firmware/bebop-linux/config/bebop_v2.yaml) |
| Per-joint torque cap (`effort_limit_sim`) | `FW_*_TAU_MAX` in `bebop_v2_base_cfg.py` | `joints.<joint>.hard_limits.tau_max` in `bebop_v2.yaml` |
| Per-joint velocity cap (`velocity_limit_sim`) | `FW_*_VEL_MAX` in `bebop_v2_base_cfg.py` | `joints.<joint>.hard_limits.vel_max` in `bebop_v2.yaml` |
| Setpoint slew on position channel (rad / 100 Hz tick) | `FW_MAX_POS_STEP_PER_TICK_RAD` in `bebop_v2_base_cfg.py`, applied via [`VariableImpedanceJointAction`](bebop_training/envs/bebop_v2_actions.py) | `defaults.slew.max_pos_step_per_tick` in `bebop_v2.yaml`, enforced in [`safe_send_ctrl`](../firmware/bebop-linux/src/safety/supervisor.rs) |
| Action / actuation latency | `FW_ACTION_DELAY_STEPS` ticks of action delay in `VariableImpedanceJointAction` | one CAN round-trip @ 100 Hz tokio tick in [`policy_runner.rs`](../firmware/bebop-linux/src/policy_runner.rs) |
| 52-dim observation / 24-dim MIT-mode action layout | [`ObservationsCfg`](bebop_training/envs/bebop_v2_base_cfg.py) | [`observation.rs`](../firmware/bebop-linux/src/observation.rs) and [`policy_runner.rs`](../firmware/bebop-linux/src/policy_runner.rs) |

If you tune any of those numbers (e.g. raise `max_pos_step_per_tick` to
let the policy move faster, or widen a `policy_gain_clamps` range after
a motor swap), update **both** sides and **retrain** — the policy bakes
in the achievable bandwidth and the kp/kd envelope, and a checkpoint
trained against one slew rate / clamp set will not transfer to another.

## Training

Training runs **inside the Isaac Lab container** (`bebop_isaac_lab`)
using [`train_bebop.py`](train_bebop.py) — a thin rsl_rl PPO entry
point that lives in this directory. At container start
`docker/entrypoint.sh` pip-installs `bebop_training` in editable mode,
which registers all the Bebop V2 Gym tasks so `--task` finds them.

Inside the container the script is at
`/workspace/bebop_bot/sim/train_bebop.py` (the repo is bind-mounted at
`/workspace/bebop_bot/`).

### Get a shell

```sh
just lab-up                   # bring up Isaac Lab + bebop_ros2 (profile: lab)
just lab-shell                # exec into bebop_isaac_lab
cd /workspace/bebop_bot/sim   # where train_bebop.py + bebop_training/ live
```

### Registered tasks

These IDs come from [`bebop_training/__init__.py`](bebop_training/__init__.py):

| Task ID                          | Cfg                            | What it trains                                                       |
|----------------------------------|--------------------------------|----------------------------------------------------------------------|
| `Isaac-BebopV2-Flat-v0`          | `BebopV2FlatBalanceCfg`        | Stand in place AND recover from random pushes (merged task).         |
| `Isaac-BebopV2-Locomotion-v0`    | `BebopV2FlatLocomotionCfg`     | Velocity tracking with light pushes.                                 |
| `Isaac-Bebop-Flat-v0`            | `BebopFlatBalanceCfg`          | Legacy V1 stand-only task (kept for reference).                      |

### Retraining after a sim-to-real fix

If you've just changed any of the `FW_*` constants in
`bebop_v2_base_cfg.py` (or the matching firmware fields in
`bebop_v2.yaml`), the existing checkpoints under `sim/logs/rsl_rl/` are
stale — they were trained against a different control surface and will
collapse on hardware. Retrain the curriculum from scratch (do not
`--resume` an old checkpoint that pre-dates the fix; it will hurt more
than it helps because the actor's converged action distribution is tuned
to the old physics).

The first stage (`Isaac-BebopV2-Flat-v0`) runs in ~45 min on a single
RTX 5090 at `--num_envs 4096`; the full curriculum is ~3 hours.

### Curriculum (recommended)

Train in sequence, warm-starting locomotion from the balance checkpoint.
The balance task is now a single merged stage that trains both standing
*and* push recovery (initial-condition randomisation + periodic lateral
pushes are wired into `BebopV2BaseEnvCfg.EventCfg`), so a separate
"robust" stage is no longer needed. See the long-form discussion in
[`experiments/exp_flat_balance_v2.py`](bebop_training/experiments/exp_flat_balance_v2.py)
and
[`experiments/exp_flat_locomotion_v2.py`](bebop_training/experiments/exp_flat_locomotion_v2.py).

```sh
# Stage 1 — stand + push recovery (single merged task)
/workspace/isaaclab/isaaclab.sh -p train_bebop.py \
    --task Isaac-BebopV2-Flat-v0 \
    --num_envs 512 --seed 42 --visualizer newton

# Stage 2 — locomotion (warm-start from stage 1)
/workspace/isaaclab/isaaclab.sh -p train_bebop.py \
    --task Isaac-BebopV2-Locomotion-v0 \
    --num_envs 512 --seed 42 --visualizer newton \
    --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<run>/model_4000.pt \
    --reset_action_std 0.5
```

Substitute the actual run timestamp under `logs/rsl_rl/<task>/` for
`<run>` (rsl_rl auto-creates a timestamped subdir per run). Pick a
checkpoint that has converged — `model_4000.pt` is a reasonable
default for the balance stages; check the TensorBoard reward curves
and use a later one if it's still improving.

### Useful flags

| Flag                       | Purpose                                                                    |
|----------------------------|----------------------------------------------------------------------------|
| `--task`                   | Gym ID — see "Registered tasks" above.                                     |
| `--num_envs`               | Parallel envs (512 is a good default on a single 4090; scale to GPU mem).  |
| `--seed`                   | RNG seed for reproducibility.                                              |
| `--visualizer newton`      | Use the Newton renderer (faster than the default Kit viewport for headless training). |
| `--resume <path>`          | Warm-start from an existing checkpoint (used for curriculum chaining).     |
| `--reset_action_std 0.5`   | Re-inject exploration noise after `--resume`. Without this, a converged policy's tiny action std stays tiny across the curriculum boundary and PPO can't escape the previous stage's local optimum. |
| `--headless`               | Disable the GUI (default for batch training; omit if you want to watch).   |

### Logs and checkpoints

`rsl_rl` writes logs and checkpoints to `logs/rsl_rl/<task>/<run>/`,
which is bind-mounted onto the host at `sim/logs/` (see the `isaac_lab`
service in `docker-compose.yml`). Inspect with TensorBoard:

```sh
# On the host:
tensorboard --logdir sim/logs/rsl_rl
```

### Exporting a trained policy

After Stage 3 converges, export the policy to ONNX for the on-robot
runner ([`bebop_pilot/policy_runner`](../ros2/src/bebop_pilot/)):

```sh
/workspace/isaaclab/isaaclab.sh -p \
    /workspace/bebop_bot/sim/bebop_training/export_bebop_model.py \
    --task Isaac-BebopV2-Locomotion-v0 \
    --checkpoint logs/rsl_rl/Isaac-BebopV2-Locomotion-v0/<run>/model_<N>.pt
```

The resulting `policy.onnx` is what `bebop_pilot bringup.launch.py`
consumes via the `model_path:=...` launch argument on the Jetson.

## DDS / ROS 2 bridge

Both Isaac containers and the ROS 2 dev container share
`ROS_DOMAIN_ID=0` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` so topics
flow between them on the same host.
