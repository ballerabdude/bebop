#!/usr/bin/env bash
#
# Installs + enables `bebop-app.service`, which starts the companion app's
# Vite dev server on boot (served on 0.0.0.0:1420; reachable at
# http://bebop.local:1420 or the robot's IP).
#
# Usage (on the robot, from the repo root):
#   sudo ./bebop-app/deploy/install-app-dev.sh
#
# Assumes the repo lives at ~/bebop for the `bebop` user and that
# `npm install` has already run in bebop-app/.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "install-app-dev.sh must be run as root (sudo)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_DIR="${REPO_DIR}/bebop-app"
UNIT_SRC="${APP_DIR}/deploy/systemd/bebop-app.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
    echo "missing ${UNIT_SRC}" >&2
    exit 1
fi
if [[ ! -d "${APP_DIR}/node_modules" ]]; then
    echo "WARN: ${APP_DIR}/node_modules is missing; run 'npm install' first" >&2
fi

chmod +x "${APP_DIR}/deploy/dev-server.sh"
install -m 0644 "${UNIT_SRC}" /etc/systemd/system/bebop-app.service
systemctl daemon-reload
systemctl enable --now bebop-app.service
systemctl --no-pager --lines=0 status bebop-app.service || true

echo
echo "The app is on http://bebop.local:1420 (or http://<robot-ip>:1420)."
echo "Logs: journalctl -u bebop-app -f"
