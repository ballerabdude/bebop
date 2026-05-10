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

## DDS / ROS 2 bridge

Both Isaac containers and the ROS 2 dev container share
`ROS_DOMAIN_ID=0` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` so topics
flow between them on the same host.
