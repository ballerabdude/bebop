#!/usr/bin/env bash
#
# Installs bebop-agent on a Jetson device.
#
# Usage: sudo ./install.sh [path-to-bebop-agent-binary]
#
# If no binary path is provided, the script assumes
# `target/aarch64-unknown-linux-gnu/release/bebop-agent` exists in the
# jetson-agent/ workspace root (i.e. one level up from `deploy/`).

set -euo pipefail

# Two levels up from this script: deploy/scripts/ -> deploy/ -> jetson-agent/
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_SRC="${1:-${WORKSPACE_ROOT}/target/aarch64-unknown-linux-gnu/release/bebop-agent}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "install.sh must be run as root (sudo)" >&2
    exit 1
fi

if [[ ! -f "${BIN_SRC}" ]]; then
    echo "bebop-agent binary not found at: ${BIN_SRC}" >&2
    exit 1
fi

echo "==> installing bebop-agent to /usr/local/bin"
install -m 0755 "${BIN_SRC}" /usr/local/bin/bebop-agent

echo "==> creating /etc/bebop"
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
