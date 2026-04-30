#!/bin/bash
# Script to download and prepare the Jetson Linux (L4T) environment for flashing
# Target: Jetson Orin Nano Super Developer Kit
# OS: Ubuntu 22.04 (L4T 36.4.4 / JetPack 6.x)

set -e

# Define URLs for L4T 36.4.4 (JetPack 6.x)
L4T_DRIVER_URL="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Jetson_Linux_r36.4.4_aarch64.tbz2"
L4T_ROOTFS_URL="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Tegra_Linux_Sample-Root-Filesystem_r36.4.4_aarch64.tbz2"

L4T_DRIVER_FILE="Jetson_Linux_r36.4.4_aarch64.tbz2"
L4T_ROOTFS_FILE="Tegra_Linux_Sample-Root-Filesystem_r36.4.4_aarch64.tbz2"

echo "================================================="
echo " Jetson L4T 36.4.4 Environment Setup Script"
echo "================================================="

# 1. Download Driver Package
if [ ! -f "$L4T_DRIVER_FILE" ]; then
    echo "[1/5] Downloading L4T Driver Package..."
    wget -O "$L4T_DRIVER_FILE" "$L4T_DRIVER_URL"
else
    echo "[1/5] Driver Package already downloaded."
fi

# 2. Download Sample Root Filesystem
if [ ! -f "$L4T_ROOTFS_FILE" ]; then
    echo "[2/5] Downloading Sample Root Filesystem..."
    wget -O "$L4T_ROOTFS_FILE" "$L4T_ROOTFS_URL"
else
    echo "[2/5] Sample Root Filesystem already downloaded."
fi

# 3. Extract Driver Package
echo "[3/5] Extracting L4T Driver Package (creates Linux_for_Tegra directory)..."
tar xf "$L4T_DRIVER_FILE"

# 4. Extract Sample Root Filesystem
echo "[4/5] Extracting Sample Root Filesystem into Linux_for_Tegra/rootfs..."
cd Linux_for_Tegra/rootfs/
sudo tar xpf ../../"$L4T_ROOTFS_FILE"
cd ..

# 5. Apply NVIDIA binaries to the rootfs
echo "[5/5] Applying NVIDIA binaries to the root filesystem..."
sudo ./apply_binaries.sh

echo "================================================="
echo " Setup Complete!"
echo " Next steps:"
echo " 1. Create a default user by running:"
echo "    sudo ./tools/l4t_create_default_user.sh -u <username> -p <password> -a -n <hostname>"
echo " 2. Put your Jetson into Recovery Mode and connect it via USB."
echo " 3. Flash the device using the instructions in the README."
echo "================================================="
