#!/bin/bash
# Script to download and prepare the Jetson Linux (L4T) environment for flashing
# Target: Jetson Orin Nano Super Developer Kit
#
# Supported releases:
#   39.2.0  (JetPack 7.2,  Ubuntu 24.04, kernel 6.8)  <- default
#   36.4.4  (JetPack 6.2.1, Ubuntu 22.04, kernel 5.15)
#
# Usage:
#   ./setup_env.sh [L4T_VERSION]
#   L4T_VERSION=36.4.4 ./setup_env.sh
#
# The version may be passed as the first argument or via the L4T_VERSION env var.

set -euo pipefail

# Default to the latest supported release (JetPack 7.2 / Jetson Linux 39.2).
L4T_VERSION="${1:-${L4T_VERSION:-39.2.0}}"

# Resolve the per-release download URLs. NVIDIA's release directory naming
# follows: r<major>_release_v<minor>.<patch>/release/
case "$L4T_VERSION" in
    39.2.0)
        L4T_RELEASE_DIR="r39_release_v2.0"
        ;;
    36.4.4)
        L4T_RELEASE_DIR="r36_release_v4.4"
        ;;
    *)
        echo "ERROR: Unsupported L4T_VERSION '$L4T_VERSION'." >&2
        echo "       Supported versions: 39.2.0, 36.4.4" >&2
        exit 1
        ;;
esac

L4T_BASE_URL="https://developer.nvidia.com/downloads/embedded/l4t/${L4T_RELEASE_DIR}/release"

L4T_DRIVER_FILE="Jetson_Linux_r${L4T_VERSION}_aarch64.tbz2"
L4T_ROOTFS_FILE="Tegra_Linux_Sample-Root-Filesystem_r${L4T_VERSION}_aarch64.tbz2"

L4T_DRIVER_URL="${L4T_BASE_URL}/${L4T_DRIVER_FILE}"
L4T_ROOTFS_URL="${L4T_BASE_URL}/${L4T_ROOTFS_FILE}"

echo "================================================="
echo " Jetson L4T ${L4T_VERSION} Environment Setup Script"
echo "================================================="

# 1. Download Driver Package
if [ ! -f "$L4T_DRIVER_FILE" ]; then
    echo "[1/6] Downloading L4T Driver Package..."
    wget -O "$L4T_DRIVER_FILE" "$L4T_DRIVER_URL"
else
    echo "[1/6] Driver Package already downloaded."
fi

# 2. Download Sample Root Filesystem
if [ ! -f "$L4T_ROOTFS_FILE" ]; then
    echo "[2/6] Downloading Sample Root Filesystem..."
    wget -O "$L4T_ROOTFS_FILE" "$L4T_ROOTFS_URL"
else
    echo "[2/6] Sample Root Filesystem already downloaded."
fi

# 3. Extract Driver Package
echo "[3/6] Extracting L4T Driver Package (creates Linux_for_Tegra directory)..."
tar xf "$L4T_DRIVER_FILE"

# 4. Extract Sample Root Filesystem
echo "[4/6] Extracting Sample Root Filesystem into Linux_for_Tegra/rootfs..."
cd Linux_for_Tegra/rootfs/
sudo tar xpf ../../"$L4T_ROOTFS_FILE"
cd ..

# 5. Install host-side flashing prerequisites
# Required since JetPack 7.x; harmless (and present) on JetPack 6.x.
if [ -f "./tools/l4t_flash_prerequisites.sh" ]; then
    echo "[5/6] Installing host flashing prerequisites..."
    sudo ./tools/l4t_flash_prerequisites.sh
else
    echo "[5/6] l4t_flash_prerequisites.sh not found; skipping."
fi

# 6. Apply NVIDIA binaries to the rootfs
echo "[6/6] Applying NVIDIA binaries to the root filesystem..."
sudo ./apply_binaries.sh

echo "================================================="
echo " Setup Complete! (L4T ${L4T_VERSION})"
echo " Next steps:"
echo " 1. Create a default user by running:"
echo "    sudo ./tools/l4t_create_default_user.sh -u <username> -p <password> -a -n <hostname>"
echo " 2. Put your Jetson into Recovery Mode and connect it via USB."
echo " 3. Flash the device using the instructions in the README."
echo "================================================="
