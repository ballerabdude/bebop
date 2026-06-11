#!/usr/bin/env bash
#
# Pull actuator system-id CSV logs off the robot into sim/bebop-sysid-logs/
# using STABLE, timestamp-free filenames so each new run REPLACES the previous
# log for the same joint+maneuver instead of accumulating timestamped copies.
#
# The sysid binary names its output:
#     sysid_<joint>_<maneuver>_<YYYYMMDD>_<HHMMSS>.csv
# This script strips the trailing "_<date>_<time>" so it lands as:
#     sysid_<joint>_<maneuver>.csv
# When several timestamped runs of the same joint+maneuver exist, the most
# recent one wins (files are processed in ascending timestamp order).
#
# Usage:
#     sim/scripts/sync_sysid_logs.sh            # scp from robot, then normalize
#     sim/scripts/sync_sysid_logs.sh --local    # only normalize files already in DEST
#
# Override defaults via env vars:
#     SYSID_REMOTE=bebop@bebop.local:~/bebop-sysid-logs
#     SYSID_DEST=/path/to/sim/bebop-sysid-logs
#
set -euo pipefail

REMOTE="${SYSID_REMOTE:-bebop@bebop.local:~/bebop-sysid-logs}"
DEST="${SYSID_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bebop-sysid-logs}"

PULL=1
if [[ "${1:-}" == "--local" || "${1:-}" == "--no-pull" ]]; then
    PULL=0
fi

mkdir -p "$DEST"

if [[ "$PULL" -eq 1 ]]; then
    echo "Pulling CSV logs from ${REMOTE} ..."
    # Copy into DEST first (timestamped), then collapse to stable names below.
    scp "${REMOTE}/"*.csv "$DEST/"
fi

echo "Normalizing filenames in ${DEST} ..."
shopt -s nullglob
renamed=0
for f in "$DEST"/sysid_*_*.csv; do
    base="$(basename "$f")"
    stable="$(printf '%s' "$base" | sed -E 's/_[0-9]{8}_[0-9]{6}\.csv$/.csv/')"
    # Skip files that don't carry a timestamp suffix (already stable).
    if [[ "$stable" == "$base" ]]; then
        continue
    fi
    mv -f "$f" "$DEST/$stable"
    echo "  ${base} -> ${stable}"
    renamed=$((renamed + 1))
done

if [[ "$renamed" -eq 0 ]]; then
    echo "  (nothing to rename — all logs already use stable names)"
fi

echo "Done. ${DEST} now holds one CSV per joint+maneuver:"
ls -1 "$DEST"/*.csv 2>/dev/null | sed 's#^#  #' || true
