#!/usr/bin/env bash
#
# Builds the `gs_usb` kernel module out-of-tree and installs it under
# /lib/modules/$(uname -r)/extra/. NVIDIA's stock JetPack 6 kernel
# (5.15.148-tegra at the time of writing) ships *without*
# CONFIG_CAN_GS_USB, so the candleLight / Geschwister Schneider
# USB-CAN hub the robot uses (USB id 1d50:606f) has no driver until
# this script runs once.
#
# Idempotent: bails out cleanly if `gs_usb` is already loaded or
# already installed somewhere modprobe can find it.
#
# Usage:
#   sudo ./build-gs-usb.sh

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "build-gs-usb.sh must be run as root (sudo)" >&2
    exit 1
fi

KVER="$(uname -r)"
echo "==> running kernel: ${KVER}"

# ---------------------------------------------------------------------------
# Fast paths: nothing to do if the module is already present.
# ---------------------------------------------------------------------------

if lsmod | awk '{print $1}' | grep -qx gs_usb; then
    echo "    gs_usb already loaded — done."
    exit 0
fi

if modinfo gs_usb >/dev/null 2>&1; then
    echo "    gs_usb already installed under /lib/modules/${KVER}; loading it"
    modprobe gs_usb
    exit 0
fi

# ---------------------------------------------------------------------------
# Toolchain + kernel headers
# ---------------------------------------------------------------------------

if ! command -v apt-get >/dev/null 2>&1; then
    echo "this script assumes a Debian/Ubuntu Jetson; bailing" >&2
    exit 1
fi

echo "==> ensuring build deps + kernel headers are installed"
DEBIAN_FRONTEND=noninteractive apt-get update -qq

apt_install_if_missing() {
    local missing=()
    for pkg in "$@"; do
        if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
            missing+=("${pkg}")
        fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
        return 0
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}

# Generic toolchain.
apt_install_if_missing build-essential bc curl

# Kernel headers: stock Ubuntu uses `linux-headers-<kver>`; JetPack ships
# `nvidia-l4t-kernel-headers` instead (which lays down the headers
# *and* the /lib/modules/$(uname -r)/build symlink).
HEADERS_PKG=""
for candidate in \
    "linux-headers-${KVER}" \
    "nvidia-l4t-kernel-headers"; do
    if apt-cache show "${candidate}" >/dev/null 2>&1; then
        HEADERS_PKG="${candidate}"
        break
    fi
done

if [[ -z "${HEADERS_PKG}" ]]; then
    echo "could not find a matching kernel-headers package in apt." >&2
    echo "Install kernel headers for ${KVER} manually, then re-run this script." >&2
    exit 1
fi
echo "    headers package: ${HEADERS_PKG}"
apt_install_if_missing "${HEADERS_PKG}"

KBUILD="/lib/modules/${KVER}/build"
if [[ ! -d "${KBUILD}" ]]; then
    echo "${KBUILD} not found after installing ${HEADERS_PKG}." >&2
    echo "JetPack sometimes ships the headers but not the symlink. Check" >&2
    echo "/usr/src and create the symlink manually if needed." >&2
    exit 1
fi
echo "    kernel build dir: ${KBUILD}"

# ---------------------------------------------------------------------------
# Fetch gs_usb.c that matches the running kernel
# ---------------------------------------------------------------------------

# 5.15.148-tegra → 5.15.148 → 5.15
KBASE_FULL="$(echo "${KVER}" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' | head -n1)"
KBASE_MAJOR="$(echo "${KBASE_FULL}" | cut -d. -f1-2)"
echo "==> kernel base version: ${KBASE_FULL} (${KBASE_MAJOR}.y)"

WORK="$(mktemp -d -t gs_usb-build.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

# Try, in order:
#   1. exact stable tag on the linux-stable mirror
#   2. matching stable branch (e.g. linux-5.15.y) — usually ahead of the
#      running kernel but the gs_usb.c API is stable across patch releases
#   3. mainline at the matching base tag (e.g. v5.15)
URLS=(
    "https://raw.githubusercontent.com/gregkh/linux/v${KBASE_FULL}/drivers/net/can/usb/gs_usb.c"
    "https://raw.githubusercontent.com/gregkh/linux/linux-${KBASE_MAJOR}.y/drivers/net/can/usb/gs_usb.c"
    "https://raw.githubusercontent.com/torvalds/linux/v${KBASE_MAJOR}/drivers/net/can/usb/gs_usb.c"
)

GS_USB_C="${WORK}/gs_usb.c"
fetched=0
for url in "${URLS[@]}"; do
    echo "    trying ${url}"
    if curl -fsSL "${url}" -o "${GS_USB_C}"; then
        echo "    fetched"
        fetched=1
        break
    fi
done

if [[ "${fetched}" -ne 1 ]]; then
    echo "couldn't fetch gs_usb.c from any candidate URL" >&2
    exit 1
fi

cat > "${WORK}/Makefile" <<'EOF'
obj-m := gs_usb.o
KDIR  := /lib/modules/$(shell uname -r)/build
PWD   := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
EOF

# ---------------------------------------------------------------------------
# Build + install
# ---------------------------------------------------------------------------

echo "==> building gs_usb.ko"
make -C "${WORK}"

if [[ ! -f "${WORK}/gs_usb.ko" ]]; then
    echo "build finished but gs_usb.ko is missing — investigate make output above" >&2
    exit 1
fi

DST_DIR="/lib/modules/${KVER}/extra"
echo "==> installing → ${DST_DIR}/gs_usb.ko"
install -d -m 0755 "${DST_DIR}"
install -m 0644 "${WORK}/gs_usb.ko" "${DST_DIR}/gs_usb.ko"

echo "==> depmod -a ${KVER}"
depmod -a "${KVER}"

echo "==> modprobe gs_usb"
if ! modprobe gs_usb 2>/tmp/gs_usb-modprobe.err; then
    cat /tmp/gs_usb-modprobe.err >&2
    if grep -q "Required key not available" /tmp/gs_usb-modprobe.err; then
        cat >&2 <<'EOF'

The kernel rejected the unsigned module because module signature
enforcement is on. Either disable enforcement (Secure Boot off) or
sign the module with the kernel's MOK. On JetPack you typically:

    sudo apt-get install -y mokutil
    # ...generate a key, enroll it via mokutil, sign gs_usb.ko, reboot.

Search for "JetPack module signing" in NVIDIA's docs for the exact
flow on your release.
EOF
    fi
    exit 1
fi

echo
echo "Done. Verify:"
echo "  lsmod | grep gs_usb"
echo "  ip -brief link show type can"
