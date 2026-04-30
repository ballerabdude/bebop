#!/bin/bash
# Entrypoint script for Isaac Lab/Sim containers.
# Installs the bebop_training package (sim/) in editable mode.
#
# Path layout inside the container:
#   /workspace/bebop_bot/         <- repo root (bind-mounted from `.`)
#   /workspace/bebop_bot/sim/     <- pyproject.toml + bebop_training/

set -e

PKG_DIR="/workspace/bebop_bot/sim"

if [ -f "${PKG_DIR}/pyproject.toml" ]; then
    echo "[INFO] Installing bebop_training package in editable mode..."

    # Determine python command
    if [ -f "/workspace/isaaclab/isaaclab.sh" ]; then
        PYTHON_CMD="/workspace/isaaclab/isaaclab.sh -p"
    elif [ -f "/isaac-sim/python.sh" ]; then
        PYTHON_CMD="/isaac-sim/python.sh"
    else
        PYTHON_CMD="python3"
    fi

    $PYTHON_CMD -m pip install -e "${PKG_DIR}" --no-deps --quiet || {
        echo "[WARNING] Failed to install bebop_training package. Continuing anyway..."
    }
    echo "[INFO] bebop_training package installed successfully."
fi

# If no command provided, default to bash (for interactive shells)
if [ $# -eq 0 ]; then
    exec bash
else
    # Execute the provided command
    exec "$@"
fi
