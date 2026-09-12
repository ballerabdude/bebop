#!/usr/bin/env bash
#
# Installs bebop-agent on a Jetson device.
#
# Usage:
#   sudo ./install.sh [--skip-prereqs] [path-to-bebop-agent-binary]
#
# If no binary path is provided, the script assumes
# `target/release/bebop-agent` exists in the jetson-agent/ workspace root
# (i.e. one level up from `deploy/`). The agent is built natively on arm64
# now, so there's no per-target subdir.
#
# Unless `--skip-prereqs` is passed, the script will also (idempotently)
# install and enable network-manager and dbus (the agent uses `nmcli` and
# raises the Hosted Network via NetworkManager).

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
BIN_SRC="${POSITIONAL[0]:-${WORKSPACE_ROOT}/target/release/bebop-agent}"

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
        echo "     network-manager and dbus by hand)"
    else
        echo "==> ensuring system prereqs are present"
        apt_install_if_missing network-manager dbus

        echo "==> enabling system services (NetworkManager)"
        enable_unit_if_present NetworkManager.service
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
