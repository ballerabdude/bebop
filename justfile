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

# Cross-compile the agent for the Jetson (arm64 Linux). Requires `cross` and Docker.
build-jetson:
    cd jetson-agent && cross build --release --target aarch64-unknown-linux-gnu -p bebop-agent

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

# Copy a freshly cross-compiled agent + deploy tree to a robot over SSH (e.g. `just deploy user@robot.local`).
deploy HOST:
    scp jetson-agent/target/aarch64-unknown-linux-gnu/release/bebop-agent {{HOST}}:/tmp/bebop-agent
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

# --- ROS 2 dev container ---------------------------------------------------

# Build (or rebuild) only the ROS 2 dev image.
ros2-build:
    docker compose build ros2_docker

# Open an interactive shell in the running ROS 2 dev container.
ros2-shell:
    docker exec -it bebop_ros2 bash

# --- Firmware (PlatformIO) -------------------------------------------------

# Build the locomotion firmware (Teensy / embedded MCU). Requires `pio` on PATH.
fw-build TARGET="bebop-locomotion":
    cd firmware/{{TARGET}} && pio run

# Flash the locomotion firmware over USB.
fw-flash TARGET="bebop-locomotion":
    cd firmware/{{TARGET}} && pio run --target upload
