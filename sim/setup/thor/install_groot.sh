#!/usr/bin/env bash
# NVIDIA Isaac-GR00T (N1.7 VLA) installer for the Jetson AGX Thor (aarch64).
#
# This is a THIN WRAPPER around NVIDIA's upstream Thor installer — we do not
# reinvent it. Upstream ships both a bare-metal installer
# (scripts/deployment/thor/install_deps.sh) and a container
# (docker/build.sh --profile=thor). This script uses the bare-metal path and
# fixes two things upstream leaves to the user:
#   1. Isaac-GR00T ships a prebuilt flash-attn wheel via git-lfs. A plain
#      `git clone` with GIT_LFS_SKIP_SMUDGE leaves a ~134-byte pointer, and the
#      upstream installer then fails with "flash-attn has an invalid package
#      format ... end of central directory record". We `git lfs pull` first.
#   2. The NVIDIA ffmpeg 8 package omits libswscale.so.9, so torchcodec's
#      FFmpeg-8 core cannot load. Installing the FFmpeg-6 libs (libavfilter9,
#      libavdevice60) lets torchcodec fall back to core6 and decode datasets.
#
# Idempotent. Needs sudo for apt.
#
# Usage (on the Thor):   bash sim/setup/thor/install_groot.sh
set -euo pipefail

GROOT_DIR="${GROOT_DIR:-$HOME/isaac-groot}"
GROOT_REF="${GROOT_REF:-main}"
GROOT_REPO="${GROOT_REPO:-https://github.com/NVIDIA/Isaac-GR00T.git}"

ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || { echo "ERROR: Thor/aarch64 only (got $ARCH)." >&2; exit 1; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> [1/4] ensure git-lfs"
command -v git-lfs >/dev/null || $SUDO apt-get install -y --no-install-recommends git-lfs
git lfs install >/dev/null 2>&1 || true

echo "==> [2/4] clone/refresh Isaac-GR00T @ $GROOT_REF"
if [ ! -d "$GROOT_DIR/.git" ]; then
  # Fetch LFS lazily; we pull the one wheel we need in step 3.
  GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules "$GROOT_REPO" "$GROOT_DIR"
fi
git -C "$GROOT_DIR" fetch --all --tags --quiet
git -C "$GROOT_DIR" checkout -q "$GROOT_REF"
git -C "$GROOT_DIR" submodule update --init --recursive

echo "==> [3/4] fetch the prebuilt flash-attn wheel (git-lfs)"
git -C "$GROOT_DIR" lfs pull --include="scripts/deployment/jetson/wheels/*"

echo "==> [4/4] run NVIDIA's upstream Thor installer"
( cd "$GROOT_DIR" && bash scripts/deployment/thor/install_deps.sh )

echo "==> torchcodec/FFmpeg fix: install FFmpeg-6 libs the NVIDIA ffmpeg 8 pkg omits"
$SUDO apt-get install -y --no-install-recommends libavfilter9 libavdevice60

cat <<EOF

Done. Activate and verify:

  source $GROOT_DIR/.venv/bin/activate
  source $GROOT_DIR/scripts/activate_thor.sh
  python -c "import torch, flash_attn, torchcodec; print(torch.__version__, torchcodec.__version__)"

Hugging Face: the base checkpoint and its VLM backbone are gated. Log in with
an account that has accepted the licenses for nvidia/GR00T-N1.7-3B and
nvidia/Cosmos-Reason2-2B:

  hf auth login        # or: export HF_TOKEN=hf_...
EOF
