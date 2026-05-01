#!/usr/bin/env bash
#
# Installs bebop-agent on a Jetson device.
#
# Usage:
#   sudo ./install.sh [--skip-prereqs] [path-to-bebop-agent-binary]
#
# If no binary path is provided, the script assumes
# `target/aarch64-unknown-linux-gnu/release/bebop-agent` exists in the
# jetson-agent/ workspace root (i.e. one level up from `deploy/`).
#
# Unless `--skip-prereqs` is passed, the script will also (idempotently)
# install and enable: bluez, network-manager, dbus, and docker. It will
# additionally probe for `nvidia-container-toolkit` and print remediation
# instructions if missing (it does not auto-add NVIDIA's apt repo, since
# that is JetPack-version-specific).

set -euo pipefail

SKIP_PREREQS=0
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --skip-prereqs)
            SKIP_PREREQS=1
            ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        --*)
            echo "unknown flag: ${arg}" >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("${arg}")
            ;;
    esac
done

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_SRC="${POSITIONAL[0]:-${WORKSPACE_ROOT}/target/aarch64-unknown-linux-gnu/release/bebop-agent}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "install.sh must be run as root (sudo)" >&2
    exit 1
fi

if [[ ! -f "${BIN_SRC}" ]]; then
    echo "bebop-agent binary not found at: ${BIN_SRC}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Prereqs
# ---------------------------------------------------------------------------

apt_install_if_missing() {
    local missing=()
    for pkg in "$@"; do
        if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
            missing+=("${pkg}")
        fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "    already installed: $*"
        return 0
    fi
    echo "    installing: ${missing[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}

enable_unit_if_present() {
    local unit="$1"
    if systemctl list-unit-files "${unit}" >/dev/null 2>&1 \
        && systemctl list-unit-files "${unit}" | grep -q "${unit}"; then
        systemctl enable --now "${unit}" >/dev/null 2>&1 || true
    fi
}

if [[ "${SKIP_PREREQS}" -eq 0 ]]; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "==> non-Debian system detected; skipping prereq install"
        echo "    (re-run with --skip-prereqs to silence this, and install"
        echo "     bluez, network-manager, dbus, docker, and nvidia-container-toolkit by hand)"
    else
        echo "==> ensuring system prereqs are present"
        apt_install_if_missing bluez network-manager dbus

        if ! command -v docker >/dev/null 2>&1; then
            echo "    docker not found; installing docker.io from distro repo"
            apt_install_if_missing docker.io
        else
            echo "    already installed: docker ($(docker --version 2>/dev/null || echo unknown))"
        fi

        echo "==> enabling system services (bluetooth, NetworkManager, docker)"
        enable_unit_if_present bluetooth.service
        enable_unit_if_present NetworkManager.service
        enable_unit_if_present docker.service

        if ! command -v nvidia-ctk >/dev/null 2>&1 \
            && ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
            echo
            echo "WARN: nvidia-container-toolkit not detected."
            echo "      The agent can still start, but the robot-app container will"
            echo "      not get GPU access until you install it. On JetPack:"
            echo "        sudo apt-get install -y nvidia-container-toolkit"
            echo "        sudo nvidia-ctk runtime configure --runtime=docker"
            echo "        sudo systemctl restart docker"
            echo
        fi
    fi
else
    echo "==> --skip-prereqs set; not touching system packages"
fi

# ---------------------------------------------------------------------------
# Agent install
# ---------------------------------------------------------------------------

echo "==> installing bebop-agent to /usr/local/bin"
install -m 0755 "${BIN_SRC}" /usr/local/bin/bebop-agent

echo "==> creating /etc/bebop and /var/lib/bebop"
install -d -m 0755 /etc/bebop
install -d -m 0755 /var/lib/bebop

if [[ ! -f /etc/bebop/agent.toml ]]; then
    echo "==> writing default /etc/bebop/agent.toml"
    install -m 0644 "${WORKSPACE_ROOT}/deploy/examples/agent.toml" /etc/bebop/agent.toml
else
    echo "==> /etc/bebop/agent.toml exists, leaving as-is"
fi

echo "==> installing systemd unit"
install -m 0644 \
    "${WORKSPACE_ROOT}/deploy/systemd/bebop-agent.service" \
    /etc/systemd/system/bebop-agent.service

echo "==> reloading systemd and enabling bebop-agent"
systemctl daemon-reload
systemctl enable --now bebop-agent.service

echo
echo "Done. Check status with: systemctl status bebop-agent"
echo "Tail logs with:         journalctl -u bebop-agent -f"
