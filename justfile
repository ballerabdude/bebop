# Common dev commands. Install `just` from https://github.com/casey/just
# then run `just` to see this list, or `just <name>` to execute one.

# Pass recipe args to shebang scripts as $1, $2, … (just 1.0+). Without this,
# `lab-export <ckpt>` silently ignored its argument and always fell back to the
# latest checkpoint, because shebang recipes don't expose "$@" by default.
set positional-arguments := true

default:
    @just --list

# --- Rust / agent ----------------------------------------------------------

# Full workspace check on the host (stubs BLE on non-Linux).
check:
    cd jetson-agent && cargo check --workspace --all-targets

# Run all unit tests.
test:
    cd jetson-agent && cargo test --workspace

# Format the workspace with rustfmt.
fmt:
    cd jetson-agent && cargo fmt --all

# Lint the workspace with clippy (warnings = errors).
lint:
    cd jetson-agent && cargo clippy --workspace --all-targets -- -D warnings

# Build the agent for the Jetson. Native build — must run on an arm64 Linux
# host (the robot itself, an arm64 dev box, or a CI `ubuntu-22.04-arm` runner).
# From an x86 host, grab the `bebop-agent-aarch64` artifact from CI instead.
build-jetson:
    cd jetson-agent && cargo build --release -p bebop-agent

# --- Robot app container ---------------------------------------------------

APP_IMAGE := env_var_or_default("APP_IMAGE", "your-registry/bebop-app:dev")

# Build the robot application image for arm64 Jetsons.
build-app:
    docker buildx build \
        --platform linux/arm64 \
        -t {{APP_IMAGE}} \
        -f jetson-agent/robot-app/Dockerfile \
        jetson-agent/robot-app

push-app:
    docker push {{APP_IMAGE}}

# --- Install on a robot ----------------------------------------------------

# Copy a freshly built agent + deploy tree to a robot over SSH (e.g. `just deploy user@robot.local`).
# Assumes you've already run `just build-jetson` on an arm64 host.
deploy HOST:
    scp jetson-agent/target/release/bebop-agent {{HOST}}:/tmp/bebop-agent
    rsync -a jetson-agent/deploy/ {{HOST}}:/tmp/deploy/
    ssh {{HOST}} 'sudo /tmp/deploy/scripts/install.sh /tmp/bebop-agent'

# --- Mobile app ------------------------------------------------------------

# Run the companion app in Tauri dev mode (desktop).
app-dev:
    cd bebop-app && npm run tauri dev

# Run the React UI in a browser (Web Bluetooth transport).
app-web:
    cd bebop-app && npm run dev

# --- Sim / training (Isaac Sim + Isaac Lab) --------------------------------

# TensorBoard for rsl_rl logs (run from repo root; opens sim/logs/rsl_rl).
tb:
    tensorboard --logdir sim/logs/rsl_rl

# Copy-friendly run list with play/resume commands (companion to TensorBoard).
tb-runs PORT="6007":
    cd sim && python scripts/training_runs_server.py --port {{PORT}}

# Bring up Isaac Sim + the ROS 2 dev container (profile: sim).
sim-up:
    docker compose --profile sim up --build -d

# Tear down the sim profile.
sim-down:
    docker compose --profile sim down

# Bring up Isaac Lab + the ROS 2 dev container (profile: lab).
lab-up:
    docker compose --profile lab up --build -d

# Tear down the lab profile.
lab-down:
    docker compose --profile lab down

# Launch the Isaac Sim GUI inside the running bebop_isaac_sim container.
# Requires the sim profile to be up (`just sim-up`) and X11 forwarding on
# the host; `xhost +local:docker` is run best-effort so windows can open.
sim-launch:
    @xhost +local:docker >/dev/null 2>&1 || true
    docker exec -it bebop_isaac_sim /isaac-sim/isaac-sim.sh

# Isaac Sim is launchable from inside this container via
# `/workspace/isaaclab/isaaclab.sh -s`, so no separate sim-shell is needed.
#
# Open an interactive shell in the running Isaac Lab container.
lab-shell:
    docker exec -it bebop_isaac_lab bash

# Launch a training run DETACHED inside the lab container so Ctrl+C in your
# shell doesn't fight Isaac Sim's SIGINT handler (which swallows the first
# interrupt during a 10-60s graceful teardown and looks like a hang).
#
# The run's stdout/stderr go to /tmp/train_<timestamp>.log inside the
# container. Run `just lab-train-ps` to see live PIDs + log paths, and
# `just lab-train-kill <pid>` to stop a specific run without touching the
# container or any sibling runs.
#
# Examples:
#   just lab-train                                      # defaults
#   just lab-train --num_envs 4096 --max_iterations 5000
#   just lab-train --task Isaac-BebopV2-Standing-Push-v0 --resume <ckpt>
lab-train *ARGS:
    #!/usr/bin/env bash
    set -e
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="/tmp/train_${TS}.log"
    # Default to the Standing task only if the caller didn't already pass --task.
    # just interpolates {{ARGS}} as a single string into the bash script, so we
    # match against that literal string.
    case " {{ARGS}} " in
        *" --task "*) TASK="" ;;
        *) TASK="--task Isaac-BebopV2-Standing-v0" ;;
    esac
    echo "[lab-train] starting; log: ${LOG}"
    echo "[lab-train] tail with:  just lab-train-log ${LOG}"
    echo "[lab-train] list runs:  just lab-train-ps"
    docker exec -d -w /workspace/bebop_bot/sim bebop_isaac_lab bash -lc \
        "/workspace/isaaclab/isaaclab.sh -p train_bebop.py ${TASK} {{ARGS}} > ${LOG} 2>&1"
    # print the PID so the user can kill it directly if needed
    sleep 1
    docker exec bebop_isaac_lab bash -lc \
        'pgrep -af "train_bebop.py" | grep -v pgrep | tail -1' || true

# List live training runs inside the lab container (PID, elapsed, log path).
# Shows only the leaf python3 process per run (each run spawns a 5-deep
# wrapper chain via isaaclab.sh -> python.sh -> python3 -> python.sh ->
# python3; the leaf is the actual training process, the rest are wrappers
# that exit automatically when the leaf dies). The log path is read from
# the process's stdout fd (/proc/<pid>/fd/1), which is the redirect target
# set by `just lab-train`.
lab-train-ps:
    #!/usr/bin/env bash
    PIDS=$(docker exec bebop_isaac_lab pgrep -f "python3 train_bebop.py")
    if [ -z "$PIDS" ]; then echo "no training runs active"; exit 0; fi
    printf "%-7s %-9s %-32s %s\n" "PID" "ELAPSED" "LOG" "CMD"
    for pid in $PIDS; do
        etime=$(docker exec bebop_isaac_lab ps -o etime= -p "$pid" | tr -d ' ')
        log=$(docker exec bebop_isaac_lab readlink "/proc/$pid/fd/1" 2>/dev/null || echo "?")
        cmd=$(docker exec bebop_isaac_lab ps -o cmd= -p "$pid" | sed 's|.*python3 train_bebop.py|train_bebop.py|')
        printf "%-7s %-9s %-32s %s\n" "$pid" "$etime" "$log" "$cmd"
    done

# Tail a training run's log (pass the path from `just lab-train-ps`, or
# omit to tail the most recent /tmp/train_*.log).
lab-train-log LOG="":
    #!/usr/bin/env bash
    if [ -z "{{LOG}}" ]; then
        LOG=$(docker exec bebop_isaac_lab bash -c 'ls -t /tmp/train_*.log 2>/dev/null | head -1')
    else
        LOG="{{LOG}}"
    fi
    if [ -z "$LOG" ]; then echo "no train logs found in /tmp"; exit 1; fi
    echo "tailing $LOG"
    docker exec bebop_isaac_lab tail -f "$LOG"

# Kill a specific training run by PID without affecting the container or
# sibling runs. Use `just lab-train-kill latest` for the most recent run,
# or `just lab-train-kill all` to stop every active run. Sends SIGTERM
# first (lets Isaac Sim flush logs), then SIGKILL after 5s if still alive.
#
# Targets only the leaf `python3 train_bebop.py` process — the 4 wrapper
# processes above it (isaaclab.sh / python.sh / the CLI bootstrap) exit
# automatically once the leaf dies.
lab-train-kill TARGET="latest":
    #!/usr/bin/env bash
    set -e
    if [ "{{TARGET}}" = "latest" ]; then
        PIDS=$(docker exec bebop_isaac_lab pgrep -f "python3 train_bebop.py" | tail -1)
    elif [ "{{TARGET}}" = "all" ]; then
        PIDS=$(docker exec bebop_isaac_lab pgrep -f "python3 train_bebop.py")
    else
        PIDS="{{TARGET}}"
    fi
    if [ -z "$PIDS" ]; then echo "no training runs to kill"; exit 0; fi
    echo "sending SIGTERM to: $PIDS"
    docker exec bebop_isaac_lab kill $PIDS 2>/dev/null || true
    sleep 5
    for p in $PIDS; do
        if docker exec bebop_isaac_lab kill -0 "$p" 2>/dev/null; then
            echo "still alive; sending SIGKILL to $p"
            docker exec bebop_isaac_lab kill -9 "$p" 2>/dev/null || true
        fi
    done
    echo "done"

# Export a trained RSL-RL checkpoint to ONNX (the artefact deployed on the
# robot). Runs export_bebop_model.py inside the lab container; the ONNX is
# written next to the checkpoint by default.
#
# CHECKPOINT may be:
#   - omitted            -> latest run's latest model_*.pt
#   - "latest"           -> same as omitted
#   - a run directory    -> latest model_*.pt inside it
#   - a full .pt path    -> used as-is
#
# Examples:
#   just lab-export                                       # latest run, latest ckpt
#   just lab-export latest                                # same
#   just lab-export logs/rsl_rl/Isaac-BebopV2-Standing-v0/2026-07-07_12-37-23/model_500.pt
#   just lab-export logs/rsl_rl/Isaac-BebopV2-Standing-v0/2026-07-07_12-37-23   # latest ckpt in run
lab-export *ARGS:
    #!/usr/bin/env bash
    set -e
    # `just` does NOT pass recipe args to shebang scripts via "$@" by default
    # — the global `set positional-arguments := true` at the top of this
    # justfile makes ARGS populate $1, $2, … so we can use the bash "$@" /
    # array machinery normally. Without it, "$@" is always empty and the
    # script silently falls back to "latest", ignoring an explicit
    # checkpoint path.
    ARGS=("$@")
    # If the first ARG is a checkpoint path or run dir, pop it off; else use "latest".
    CKPT="latest"
    PASS_ARGS=()
    if [ ${#ARGS[@]} -gt 0 ]; then
        case "${ARGS[0]}" in
            --*) ;;                    # flag -> keep CKPT=latest, pass all ARGS through
            *)  CKPT="${ARGS[0]}"; PASS_ARGS=("${ARGS[@]:1}") ;;
        esac
    fi
    # Resolve the checkpoint path inside the container.
    if [ "$CKPT" = "latest" ]; then
        CKPT=$(docker exec -w /workspace/bebop_bot/sim bebop_isaac_lab bash -c \
            'ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1')
        if [ -z "$CKPT" ]; then echo "no checkpoints found under sim/logs/rsl_rl"; exit 1; fi
        echo "[lab-export] latest checkpoint: $CKPT"
    elif echo "$CKPT" | grep -qv '\.pt$'; then
        # Treat as a run directory -> pick the latest model_*.pt inside it.
        DIR="${CKPT%/}"
        RESOLVED=$(docker exec -w /workspace/bebop_bot/sim bebop_isaac_lab bash -c \
            "ls -t '${DIR}'/model_*.pt 2>/dev/null | head -1")
        if [ -z "$RESOLVED" ]; then
            echo "no model_*.pt found in $DIR"; exit 1
        fi
        CKPT="$RESOLVED"
        echo "[lab-export] latest checkpoint in run: $CKPT"
    else
        echo "[lab-export] checkpoint: $CKPT"
    fi
    echo "[lab-export] exporting to ONNX..."
    docker exec -w /workspace/bebop_bot/sim bebop_isaac_lab bash -lc \
        "/workspace/isaaclab/isaaclab.sh -p bebop_training/export_bebop_model.py --checkpoint $CKPT ${PASS_ARGS[*]}"

# Play a trained policy with the interactive torso-push controller wired up.
# All flags are forwarded to play_bebop.py. If --task or --resume are omitted,
# sensible defaults are inserted (latest Standing-v0 run).
#
# Examples:
#     just lab-play                                               # defaults
#     just lab-play --resume sim/logs/rsl_rl/Isaac-BebopV2-Standing-v0/2026-07-13_22-30-37/model_4000.pt
#     just lab-play --task Isaac-BebopV2-Standing-FixedGain-v0 --resume <run_dir> --num_envs 1
#     just lab-play --resume <ckpt> --print_obs_actions
#
# Once the Isaac Sim window opens, CLICK INSIDE THE VIEWPORT so it receives
# keyboard focus, then use:
#     I/K = pitch nose-down / nose-up
#     J/L = roll left / right
#     W/S = push forward / backward    A/D = push left / right
#     +/- = scale impulse 1.25x        R = reset env  H = help  0 = print scale
# See sim/play_bebop.py module docstring for the full reference.
lab-play *ARGS:
    #!/usr/bin/env bash
    set -e
    # Default --task / --resume only if the caller didn't already pass them.
    # just interpolates {{ARGS}} as a single string, so we match against it.
    case " {{ARGS}} " in
        *" --task "*)   TASK_FLAG="" ;;
        *)              TASK_FLAG="--task Isaac-BebopV2-Standing-v0" ;;
    esac
    case " {{ARGS}} " in
        *" --resume "*) RESUME_FLAG="" ;;
        *)              RESUME_FLAG="--resume logs/rsl_rl/Isaac-BebopV2-Standing-v0" ;;
    esac
    xhost +local:docker >/dev/null 2>&1 || true
    docker exec -it bebop_isaac_lab bash -lc \
        "cd /workspace/bebop_bot/sim && \
         /workspace/isaaclab/isaaclab.sh -p play_bebop.py \
            ${TASK_FLAG} ${RESUME_FLAG} {{ARGS}}"

# Mirror live hardware joint + IMU pose in Isaac Lab. Streams telemetry from
# bebop-linux over WebSocket (default 9090). Requires the lab container to be
# up (`just lab-up`) and bebop-linux running on the robot.
#
# Examples:
#   just lab-mirror 192.168.0.69                        # default host
#   just lab-mirror 192.168.0.69 --telemetry-hz 30      # custom rate
#   just lab-mirror 192.168.0.69 --viz newton           # Newton renderer
lab-mirror HOST *ARGS:
    @xhost +local:docker >/dev/null 2>&1 || true
    docker exec -it bebop_isaac_lab bash -lc \
        'cd /workspace/bebop_bot/sim && \
         /workspace/isaaclab/isaaclab.sh -p mirror_bebop.py \
            --robot-host {{HOST}} \
            --telemetry-hz 100 \
            {{ARGS}}'

# --- ROS 2 dev container ---------------------------------------------------

# Build (or rebuild) only the ROS 2 dev image.
ros2-build:
    docker compose build ros2_docker

# Start the ROS 2 dev container and ensure the entrypoint bootstrap has run.
ros2-up:
    docker compose --profile sim up --build -d ros2_docker
    docker exec bebop_ros2 bash -lc 'source /ros_ws_entrypoint.sh'

# Open an interactive shell in the ROS 2 dev container.
ros2-shell: ros2-up
    docker exec -it bebop_ros2 bash

# Re-expand bebopv2.xacro into bebopv2.urdf (with absolute mesh paths).
# Starts/runs inside bebop_ros2, builds just the description package (to
# avoid pulling in unrelated/broken packages like micro_ros_msgs), then
# passes extra flags through, e.g.
#   `just ros2-urdf --mesh-prefix /workspace/bebop_bot/ros2/src/bebopv2_description/meshes`
ros2-urdf *FLAGS: ros2-up
    docker exec -it bebop_ros2 bash -lc \
        'source /ros_ws_entrypoint.sh && \
         colcon build --packages-up-to bebopv2_description && \
         source install/setup.bash && \
         "$ROS_WS"/src/bebopv2_description/scripts/xacro-to-urdf.sh {{FLAGS}}'

# Launch RViz with the bebopv2_description display (robot_state_publisher +
# joint_state_publisher_gui + rviz2 preloaded with config/display.rviz).
# Requires X11 forwarding on the host; run `xhost +local:docker` once per
# login session if windows fail to open. Pass extra args through, e.g.
#   `just ros2-rviz gui:=false`
ros2-rviz *FLAGS: ros2-up
    @xhost +local:docker >/dev/null 2>&1 || true
    docker exec -it bebop_ros2 bash -lc \
        'source /ros_ws_entrypoint.sh && \
         colcon build --packages-up-to bebopv2_description && \
         source install/setup.bash && \
         ros2 launch bebopv2_description display.launch.py {{FLAGS}}'

# --- Firmware (PlatformIO) -------------------------------------------------

# Build the locomotion firmware (Teensy / embedded MCU). Requires `pio` on PATH.
fw-build TARGET="bebop-locomotion":
    cd firmware/{{TARGET}} && pio run

# Flash the locomotion firmware over USB.
fw-flash TARGET="bebop-locomotion":
    cd firmware/{{TARGET}} && pio run --target upload
