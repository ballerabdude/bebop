# sim/

Simulation, RL training, and 3D assets for Bebop. Everything in here runs
**off-robot** on a workstation with an NVIDIA GPU and Isaac Sim / Isaac Lab.

## Layout

| Path               | What it is                                                                |
|--------------------|---------------------------------------------------------------------------|
| `pyproject.toml`   | Defines the `bebop_training` Python package (depends on `isaaclab`).      |
| `bebop_training/`  | RL extension: agents, envs, experiments, model export.                    |
| `usd/`             | USD scene assets (`bebopv2/` robot description for Isaac Sim).            |
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

## DDS / ROS 2 bridge

Both Isaac containers and the ROS 2 dev container share
`ROS_DOMAIN_ID=0` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` so topics
flow between them on the same host.
