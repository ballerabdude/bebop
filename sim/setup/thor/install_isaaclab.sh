#!/usr/bin/env bash
# Pinned Isaac Sim + Isaac Lab installer for the Jetson AGX Thor (aarch64).
#
# Why a script and not a container: NVIDIA publishes no aarch64 Isaac Sim
# image (the nvcr.io isaac-sim / isaac-lab images used by sim/docker are
# x86_64), and the docs only list DGX Spark as supported aarch64. The
# pip/native path below is what actually works on Thor (JetPack 7.x / L4T
# R39, Python 3.12, CUDA 13.2). See README.md for context and gotchas.
#
# Idempotent: safe to re-run. Bump the pins deliberately, together.
#
# Usage (on the Thor):   bash sim/setup/thor/install_isaaclab.sh
set -euo pipefail

# --- pins -------------------------------------------------------------------
ISAACSIM_VERSION="${ISAACSIM_VERSION:-6.1.0.0}"                       # aarch64 cp312 wheel
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-8b29905}"                               # main, Isaac Sim 6.1 era
RL_LIB="${RL_LIB:-rsl_rl}"                                            # installs torch 2.12.0+cu130
WORKDIR="${ISAACLAB_THOR_DIR:-$HOME/isaacsim-test}"
VENV="$WORKDIR/.venv"
CUDA_HOME_VER=13.2

# --- platform check ---------------------------------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
  echo "ERROR: this recipe is for Thor/aarch64 (got $ARCH)." >&2
  echo "       On x86_64 use the containers instead: just sim-up / just lab-up" >&2
  exit 1
fi
if [ -r /etc/nv_tegra_release ]; then
  REL="$(sed -n 's/^# R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release | head -n1)"
  [ "$REL" = "39" ] || echo "WARN: expected JetPack 7.x / L4T R39, got R${REL:-unknown}." >&2
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> [1/6] system deps (ffmpeg, git-lfs, CUDA $CUDA_HOME_VER dev for ptxas)"
if [ ! -x "/usr/local/cuda-${CUDA_HOME_VER}/bin/ptxas" ]; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends \
    ffmpeg git-lfs cmake build-essential \
    "cuda-nvcc-${CUDA_HOME_VER/./-}" "cuda-cudart-dev-${CUDA_HOME_VER/./-}" "cuda-nvrtc-dev-${CUDA_HOME_VER/./-}"
else
  echo "    ptxas already present, skipping CUDA dev apt packages"
fi

echo "==> [2/6] Python 3.12 venv at $VENV"
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found" >&2; exit 1; }
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "==> [3/6] Isaac Sim $ISAACSIM_VERSION (aarch64 pip wheel)"
"$VENV/bin/pip" install \
  "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com

echo "==> [4/6] Isaac Lab @ $ISAACLAB_REF"
if [ ! -d "$WORKDIR/IsaacLab/.git" ]; then
  git clone "$ISAACLAB_REPO" "$WORKDIR/IsaacLab"
fi
git -C "$WORKDIR/IsaacLab" fetch --quiet origin "$ISAACLAB_REF" || true
git -C "$WORKDIR/IsaacLab" checkout -q "$ISAACLAB_REF"

echo "==> [5/6] Isaac Lab install ($RL_LIB) — this upgrades torch to 2.12.0+cu130"
(
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  cd "$WORKDIR/IsaacLab"
  ./isaaclab.sh -i "$RL_LIB"
)

echo "==> [6/6] environment helper -> $WORKDIR/activate_thor.sh"
cat > "$WORKDIR/activate_thor.sh" <<'EOF'
# Source this before running Isaac Lab on Thor (aarch64).
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
# Isaac Sim expects the system OpenMP; PyTorch aarch64 bundles its own libgomp.
case ":${LD_PRELOAD:-}:" in
  *":/lib/aarch64-linux-gnu/libgomp.so.1:"*) ;;
  *) export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1" ;;
esac
EOF

cat <<EOF

Done. Smoke-test the env:

  source $VENV/bin/activate
  source $WORKDIR/activate_thor.sh
  cd $WORKDIR/IsaacLab
  python scripts/reinforcement_learning/train.py \\
      --rl_library rsl_rl --task=Isaac-Ant --viz none --num_envs 256

Notes:
  * Current Isaac Lab has NO --headless flag; use --viz none.
  * Built-in task names dropped the -v0 suffix (Isaac-Ant, not Isaac-Ant-v0).
    The bebop task (Isaac-BebopV2-Standing-v0) only exists once sim/ is
    pip-installed into this venv.
  * On the robot, never train while the recorder (bebop-vision) is capturing.
EOF
