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

| Task ID                          | Cfg                            | What it trains                                  |
|----------------------------------|--------------------------------|-------------------------------------------------|
| `Isaac-BebopV2-Flat-v0`          | `BebopV2FlatBalanceCfg`        | Stand in place, no commands, no pushes.         |
| `Isaac-BebopV2-FlatRobust-v0`    | `BebopV2FlatBalanceRobustCfg`  | Stand under random pushes — learns stepping.    |
| `Isaac-BebopV2-Locomotion-v0`    | `BebopV2FlatLocomotionCfg`     | Velocity tracking with light pushes.            |
| `Isaac-Bebop-Flat-v0`            | `BebopFlatBalanceCfg`          | Legacy V1 stand-only task (kept for reference). |

### Curriculum (recommended)

Train in sequence, warm-starting each stage from the previous stage's
checkpoint. The locomotion stage **must** be warm-started from
`Isaac-BebopV2-FlatRobust-v0` (not `Isaac-BebopV2-Flat-v0`) so the
policy already knows how to step before being asked to track velocity
— see the long-form discussion in
[`experiments/exp_flat_balance_robust_v2.py`](bebop_training/experiments/exp_flat_balance_robust_v2.py)
and
[`experiments/exp_flat_locomotion_v2.py`](bebop_training/experiments/exp_flat_locomotion_v2.py).

```sh
# Stage 1 — stand only
/workspace/isaaclab/isaaclab.sh -p train_bebop.py \
    --task Isaac-BebopV2-Flat-v0 \
    --num_envs 512 --seed 42 --visualizer newton

# Stage 2 — stand under push (warm-start from stage 1)
/workspace/isaaclab/isaaclab.sh -p train_bebop.py \
    --task Isaac-BebopV2-FlatRobust-v0 \
    --num_envs 512 --seed 42 --visualizer newton \
    --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<run>/model_4000.pt \
    --reset_action_std 0.5

# Stage 3 — locomotion (warm-start from stage 2)
/workspace/isaaclab/isaaclab.sh -p train_bebop.py \
    --task Isaac-BebopV2-Locomotion-v0 \
    --num_envs 512 --seed 42 --visualizer newton \
    --resume logs/rsl_rl/Isaac-BebopV2-FlatRobust-v0/<run>/model_4000.pt \
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
