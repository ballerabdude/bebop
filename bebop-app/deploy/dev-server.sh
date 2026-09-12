#!/bin/bash
# Runs the Bebop companion app's Vite dev server on the robot, exposed on the
# network (see `vite.config.ts`: host 0.0.0.0, port 1420, allows bebop.local).
#
# This backs `deploy/systemd/bebop-app.service`. It is a development server,
# not a production deployment; node deps (`npm install`) must already exist.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$APP_DIR"

# Prefer nvm if present. `nvm use` can fail when the .nvmrc version isn't
# installed, so fall back to the newest node nvm did install.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true
fi
if ! command -v npm >/dev/null 2>&1; then
    NODE_BIN="$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1 || true)"
    if [ -n "${NODE_BIN}" ]; then
        export PATH="${NODE_BIN}:${PATH}"
    fi
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "dev-server.sh: node/npm not found (looked on PATH and under $NVM_DIR)" >&2
    exit 1
fi

exec npm run dev
