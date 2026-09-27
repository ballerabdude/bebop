#!/usr/bin/env bash
# Enter the Thor on-device Isaac Lab environment.
#
#   source sim/setup/thor/env.sh
#
# Points at the venv created by install_isaaclab.sh and sets the two env vars
# Isaac Sim needs on Thor (EULA + system libgomp preload). Safe to source twice.
#
# shellcheck shell=bash

export ISAACLAB_THOR_DIR="${ISAACLAB_THOR_DIR:-$HOME/isaacsim-test}"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

# Isaac Sim expects the system OpenMP; PyTorch aarch64 bundles its own libgomp.
case ":${LD_PRELOAD:-}:" in
  *":/lib/aarch64-linux-gnu/libgomp.so.1:"*) ;;
  *) export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1" ;;
esac

if [ -f "$ISAACLAB_THOR_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$ISAACLAB_THOR_DIR/.venv/bin/activate"
fi
