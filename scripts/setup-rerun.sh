#!/usr/bin/env bash
#
# Set up on-robot MCAP -> Rerun (.rrd) conversion.
#
# Creates a dedicated venv (pinned to the viewer's Rerun version), installs
# the converter + dashboard blueprint, and drops a wrapper the firmware
# calls from `GET /captures/rerun/<name>`.
#
# Idempotent. Run from a repo checkout on the robot:
#
#     sudo ./scripts/setup-rerun.sh
#
# The Rerun version MUST match the viewer you open the .rrd with, or the
# file won't load. Override with `RERUN_VERSION=x.y.z`.
set -euo pipefail

RERUN_VERSION="${RERUN_VERSION:-0.31.3}"
PREFIX="${PREFIX:-/opt/bebop-rerun}"
LIB_DIR="/usr/local/lib/bebop-rerun"
WRAPPER="/usr/local/bin/bebop-mcap-to-rrd"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

for f in rerun_dashboards/mcap_to_rrd.py rerun_dashboards/navd_blueprint.py \
         rerun_dashboards/system_blueprint.py; do
    if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
        echo "missing ${REPO_ROOT}/${f} (run from the repo checkout)" >&2
        exit 1
    fi
done

echo "==> venv ${PREFIX}/venv (rerun-sdk==${RERUN_VERSION})"
python3 -m venv "${PREFIX}/venv"
"${PREFIX}/venv/bin/pip" install --disable-pip-version-check -q \
    "rerun-sdk==${RERUN_VERSION}" mcap

echo "==> installing converter -> ${LIB_DIR}"
install -d -m 0755 "${LIB_DIR}"
install -m 0644 "${REPO_ROOT}/rerun_dashboards/mcap_to_rrd.py" \
    "${LIB_DIR}/mcap_to_rrd.py"
install -m 0644 "${REPO_ROOT}/rerun_dashboards/navd_blueprint.py" \
    "${LIB_DIR}/navd_blueprint.py"
install -m 0644 "${REPO_ROOT}/rerun_dashboards/system_blueprint.py" \
    "${LIB_DIR}/system_blueprint.py"

echo "==> installing wrapper -> ${WRAPPER}"
cat > "${WRAPPER}" <<EOF
#!/bin/sh
exec ${PREFIX}/venv/bin/python ${LIB_DIR}/mcap_to_rrd.py "\$@"
EOF
chmod 0755 "${WRAPPER}"

echo
echo "Done. The firmware's --rerun-converter default points at ${WRAPPER}."
echo "Verify with: ${WRAPPER} <in.mcap> /tmp/out.rrd"
