#!/usr/bin/env bash
# Removes bebop-agent from a Jetson. Leaves /var/lib/bebop intact.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "uninstall.sh must be run as root (sudo)" >&2
    exit 1
fi

systemctl disable --now bebop-agent.service || true
rm -f /etc/systemd/system/bebop-agent.service
systemctl daemon-reload

rm -f /usr/local/bin/bebop-agent

echo "bebop-agent removed. /etc/bebop and /var/lib/bebop were left in place."
