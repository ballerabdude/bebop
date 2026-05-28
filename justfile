# Common dev commands. Install `just` from https://github.com/casey/just
# then run `just` to see this list, or `just <name>` to execute one.

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

# Play a trained policy with the interactive torso-push controller wired up.
# Defaults to the standing task and the most recent run under
# sim/logs/rsl_rl/Isaac-BebopV2-Standing-v0. Override with TASK / RESUME.
#
# Once the Isaac Sim window opens, CLICK INSIDE THE VIEWPORT so it receives
# keyboard focus, then use:
#     I/K = pitch nose-down / nose-up
#     J/L = roll left / right
#     W/S = push forward / backward    A/D = push left / right
#     +/- = scale impulse 1.25x        R = reset env  H = help  0 = print scale
# See sim/play_bebop.py module docstring for the full reference.
lab-play TASK="Isaac-BebopV2-Standing-v0" RESUME="logs/rsl_rl/Isaac-BebopV2-Standing-v0":
    @xhost +local:docker >/dev/null 2>&1 || true
    docker exec -it bebop_isaac_lab bash -lc \
        'cd /workspace/bebop_bot/sim && \
         /workspace/isaaclab/isaaclab.sh -p play_bebop.py \
            --task {{TASK}} \
            --resume {{RESUME}}'

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
