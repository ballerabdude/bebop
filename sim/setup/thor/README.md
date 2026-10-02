# On-device training on the Jetson AGX Thor (aarch64)

This directory pins the *on-device* training environment for Thor. It is
deliberately **not** a container and **not** part of the robot's production
image — training is a separate, schedulable workload from `bebop-linux` /
`bebop-vision`.

## Two stacks, two strategies

| Workload | Host | Reproducibility |
|---|---|---|
| Isaac Sim / Isaac Lab (RL, sim, augmentation) | **x86_64 workstation** | `sim/docker` + `docker-compose.yml` (`just sim-up` / `just lab-up`) — unchanged |
| Isaac Sim / Isaac Lab | **Thor (aarch64)** | **pinned native venv** via `install_isaaclab.sh` (no official aarch64 image exists) |
| GR00T N1.7 VLA (imitation learning) | **Thor (aarch64)** | NVIDIA upstream: bare-metal `install_groot.sh` wrapper, or their `gr00t-thor` container |
| GR00T / VLA | **x86_64 workstation** | upstream `uv sync` / `docker/build.sh --profile=dgpu` |

**Why no Thor container for Isaac Lab:** NVIDIA's `nvcr.io/nvidia/isaac-sim`
and `isaac-lab` images (the `BASE_IMAGE`s in `sim/docker/Dockerfile`) are
x86_64-only, and the docs list only DGX Spark as a supported aarch64 target.
A hand-rolled Thor image would still bind to the host JetPack/driver for no
reproducibility gain over a venv. GR00T is different: NVIDIA ships an official
Thor container + installer, so we use theirs.

## Proven on the actual robot (2026-09)

| | |
|---|---|
| Hardware | NVIDIA Jetson AGX Thor Developer Kit |
| OS | JetPack 7.2 GA / L4T **R39.2.1** |
| CUDA / Python | CUDA **13.2**, Python **3.12** |
| GPU | `NVIDIA Thor`, compute cap **sm_110** (nm: sm_101), 122 GB unified RAM |
| Isaac Sim | 6.1.0.0 (`...-cp312-none-manylinux_2_35_aarch64.whl`) |
| Isaac Lab | `main @ 8b29905` |
| torch | 2.12.0+cu130 (arch list includes `sm_110`) |
| GR00T | N1.7 `main @ 51d4c89` ("Add support for Jetpack 7.2 on Orin and Thor") |
| Results | Isaac Lab `Isaac-Ant`, 256 envs, 5 iters ≈ 7 s. GR00T 2000-step SO100 fine-tune: 2.4 s/step, ~85 min, ~24 GB/checkpoint |

### Version pins (bump together)
| Component | Pin | Where |
|---|---|---|
| Isaac Sim | `6.1.0.0` | `install_isaaclab.sh` |
| Isaac Lab | `8b29905` | `install_isaaclab.sh` |
| torch (Isaac Lab) | `2.12.0+cu130` | pulled by `isaaclab.sh -i rsl_rl` |
| torch (GR00T) | `2.13.0+cu132` | pulled by upstream Thor installer |
| flash-attn | `2.8.3` wheel (`sm_87`+`sm_110`) | GR00T git-lfs (`scripts/deployment/jetson/wheels/`) |
| torchcodec | `0.15.0+cu132` | pulled by upstream GR00T installer |

## Isaac Lab — install and run

```sh
# On the Thor (repo at ~/bebop):
bash sim/setup/thor/install_isaaclab.sh

# Each shell:
source ~/isaacsim-test/.venv/bin/activate
source ~/isaacsim-test/activate_thor.sh      # EULA + libgomp preload
cd ~/isaacsim-test/IsaacLab
python scripts/reinforcement_learning/train.py \
    --rl_library rsl_rl --task=Isaac-Ant --viz none --num_envs 256
```

Or from the workstation via `just`:
```sh
just thor-isaaclab          # install/refresh on the robot
just thor-train --task Isaac-Ant --num_envs 256
```

`env.sh` is a repo-local equivalent of the generated `activate_thor.sh`:
`source sim/setup/thor/env.sh`.

## GR00T N1.7 — install and fine-tune

```sh
# On the Thor:
bash sim/setup/thor/install_groot.sh

source ~/isaac-groot/.venv/bin/activate
source ~/isaac-groot/scripts/activate_thor.sh
hf auth login     # accept the nvidia/GR00T-N1.7-3B + nvidia/Cosmos-Reason2-2B licenses

cd ~/isaac-groot
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 --output-dir ~/groot_runs/so100 \
    --max-steps 2000 --save-steps 500 --global-batch-size 32 --dataloader-num-workers 4
```

`just thor-groot` wraps the install.

## Known gotchas (all hit in practice)

1. **No `--headless` flag** in current Isaac Lab — use `--viz none`.
2. **Task names lost the `-v0` suffix** (`Isaac-Ant`, not `Isaac-Ant-v0`).
3. **`LD_PRELOAD` libgomp** — aarch64 PyTorch bundles `libgomp`; Isaac Sim
   wants the system one. Handled by `env.sh` / `activate_thor.sh`.
4. **PTX JIT, not SASS** — Isaac Sim's PhysX ships `sm_100`/`sm_120` cubins
   (no `sm_110`), but also `compute_100` PTX, which the driver JITs to `sm_110`.
   Verified with a standalone compute_100-PTX kernel.
5. **OptiX/NGX denoiser unavailable on Thor** (`Failed to create Optix adaptor`).
   Physics RL is fine; RTX camera/denoising workflows are limited.
6. **GR00T flash-attn wheel is git-lfs.** Cloning without LFS leaves a 134-byte
   pointer and the installer aborts with "invalid package format ... end of
   central directory record". `install_groot.sh` runs `git lfs pull` first.
7. **torchcodec needs FFmpeg 6 libs.** The NVIDIA `ffmpeg` 8.0.1-nvidia1 package
   omits `libswscale.so.9`; torchcodec's FFmpeg-8 core can't load. Installing
   `libavfilter9` + `libavdevice60` makes `core6` load (H.264 decode works).
8. **Hugging Face gating.** `nvidia/Cosmos-Reason2-2B` is `gated: auto`; the
   token's account must accept the license or loading fails 401/403.
9. **jetson-stats on Thor.** `jtop` 7.2.1 predates Thor GPU support (reports
   "GPU NOT DETECTED"). Use jetson-stats **7.2.2** from `master` until it is
   released: `sudo pip3 install --break-system-packages -U "git+https://github.com/rbonghi/jetson_stats.git"`.
   `nvidia-smi dmon` also works for live SM% even though `--query-gpu` shows N/A.
   With 7.2.2 the GPU can *still* read as not detected for the whole boot,
   because `jtop.service` and `nv-load-display-modules.service` (which loads the
   Thor nvidia driver) are both only `Before=multi-user.target` and run
   unordered: jtop's NVML probe fires ~3 s before the driver finishes, logs
   `NVML check failed: Driver Not Loaded`, and never re-probes. Order jtop after
   the driver with a drop-in:
   ```ini
   # /etc/systemd/system/jtop.service.d/10-wait-nvidia-driver.conf
   [Unit]
   After=nv-load-display-modules.service
   Wants=nv-load-display-modules.service
   ```
   then `sudo systemctl daemon-reload && sudo systemctl restart jtop` (a reboot
   is the real test). `jtop --version` will still print 7.2.2 — the version is
   not the problem once 7.2.2 is installed.

## On-device self-imitation learning (the target workflow)

The install is the easy part. A self-imitation loop needs four stages:

1. **Collect** — the existing navd MCAP recorder (`bebop-vision`, `cmd_vel` is
   the imitation label).
2. **Convert** — MCAP → training layout (`tools/mcap_extract.py` today, for the
   navd-v0 layout). Training a VLA/GR00T policy instead needs **LeRobot format
   + a modality config** — this converter is the current gap.
3. **Train** — on-device `launch_finetune.py` (proven) or the navd model.
4. **Promote** — checkpoint → ONNX / GR00T policy server → into the drive path.

Constraints to design around:
- **One camera process.** `docs/navd.md` §2.8: pyorbbecsdk capture is serial
  and the GIL is already the bottleneck — never train while the recorder runs.
- GPU contention and 120 W thermals/power; schedule training when idle.
- Disk: a GR00T 3B fine-tune checkpoint is ~24 GB each (safetensors + optimizer).

## See also
- `docs/navd.md` — recorder, MCAP channels, training layout.
- `sim/README.md` — workstation container and URDF→USD workflow.
- `sim/docker/` + `docker-compose.yml` — the x86 training environment.
