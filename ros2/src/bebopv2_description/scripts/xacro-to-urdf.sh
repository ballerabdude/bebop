#!/usr/bin/env bash
#
# Expand `bebopv2.xacro` into a flat `bebopv2.urdf` and rewrite every
# `<mesh filename="...">` to an absolute path that resolves inside the
# containers we care about.
#
# Why the rewrite step exists:
#
#   The xacro file uses `file://$(find bebopv2_description)/meshes/...`,
#   which `xacro` resolves to the package's installed *share* directory
#   (e.g. `/home/<user>/master_ros2/install/bebopv2_description/share/...`).
#   That path:
#     * doesn't exist until you've run `colcon build`,
#     * is different in the `bebop_ros2` container vs. the Isaac Sim /
#       Isaac Lab containers,
#     * is *not* resolvable by Isaac Sim's URDF→USD importer, which has
#       no ROS package index and can only follow plain `file://` URIs.
#
#   See `ros2/README.md` → "URDF mesh paths" for the long-form
#   explanation.
#
# This script:
#   1. runs `xacro` on the input file (must be run inside `bebop_ros2`,
#      where `/opt/ros/jazzy/setup.bash` and the workspace overlay are
#      available), and
#   2. rewrites all mesh URIs to `file://<MESH_PREFIX>/<basename>`.
#
# The default prefix matches what's currently committed in
# `bebopv2.urdf`. If you move the workspace mount or rename the package,
# pass `--mesh-prefix` to match.
#
# Usage (inside `bebop_ros2`, after `source /ros_ws_entrypoint.sh`):
#
#   ./scripts/xacro-to-urdf.sh
#   ./scripts/xacro-to-urdf.sh --mesh-prefix /workspace/bebop_bot/ros2/src/bebopv2_description/meshes
#   ./scripts/xacro-to-urdf.sh --in some.xacro --out some.urdf
#
# Or from the host (convenience wrapper):
#
#   just ros2-urdf
#   just ros2-urdf --mesh-prefix /workspace/bebop_bot/ros2/src/bebopv2_description/meshes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_IN="${DESC_DIR}/urdf/bebopv2.xacro"
DEFAULT_OUT="${DESC_DIR}/urdf/bebopv2.urdf"
DEFAULT_PREFIX="/workspace/bebop_bot/ros2/src/bebopv2_description/meshes"

IN="${DEFAULT_IN}"
OUT="${DEFAULT_OUT}"
MESH_PREFIX="${DEFAULT_PREFIX}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--in PATH] [--out PATH] [--mesh-prefix PATH] [--check]

Options:
  --in PATH            input xacro                (default: ${DEFAULT_IN})
  --out PATH           output URDF                (default: ${DEFAULT_OUT})
  --mesh-prefix PATH   absolute prefix to use for every <mesh filename="...">
                       (default: ${DEFAULT_PREFIX})
  --check              don't write OUT; print the rewritten URDF to stdout
  -h, --help           show this help

Notes:
  * Must be run inside the bebop_ros2 container, with the ROS overlay
    sourced (\`source /ros_ws_entrypoint.sh\`). \`xacro\` is provided by
    the ros-jazzy-xacro deb.
  * The mesh prefix should be an absolute path that exists *inside the
    container that will consume the URDF*. Both \`bebop_ros2\` and the
    Isaac containers can see the meshes; they just disagree on the
    mount point — pick the one your downstream tool expects.
EOF
}

CHECK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --in)           IN="$2";          shift 2;;
        --out)          OUT="$2";         shift 2;;
        --mesh-prefix)  MESH_PREFIX="$2"; shift 2;;
        --check)        CHECK=1;          shift;;
        -h|--help)      usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage; exit 1;;
    esac
done

if ! command -v xacro >/dev/null 2>&1; then
    echo "error: \`xacro\` not on PATH." >&2
    echo "       Run this inside bebop_ros2 after \`source /ros_ws_entrypoint.sh\`." >&2
    exit 1
fi

if [[ ! -f "${IN}" ]]; then
    echo "error: input xacro not found: ${IN}" >&2
    exit 1
fi

# Strip a trailing slash so we don't end up with `//` in the rewritten path.
MESH_PREFIX="${MESH_PREFIX%/}"

echo "==> input         : ${IN}"
echo "==> output        : ${OUT}$( ((CHECK)) && echo ' (--check, not writing)' )"
echo "==> mesh prefix   : ${MESH_PREFIX}/"

TMP_URDF="$(mktemp -t bebopv2-urdf.XXXXXX.urdf)"
trap 'rm -f "${TMP_URDF}"' EXIT

# 1. Expand xacro → flat URDF.
xacro "${IN}" -o "${TMP_URDF}"

# 2. Rewrite every <mesh filename="..."> so it points at MESH_PREFIX +
#    the original basename. We use python3 (always available in the
#    bebop_ros2 image) for safe XML-attribute rewriting; falling back to
#    sed risks mangling whitespace / quoting.
python3 - "${TMP_URDF}" "${MESH_PREFIX}" <<'PY'
import re
import sys

path, prefix = sys.argv[1], sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

mesh_re = re.compile(r'(<mesh\b[^>]*\bfilename=")([^"]+)(")')

def rewrite(match: "re.Match[str]") -> str:
    head, uri, tail = match.group(1), match.group(2), match.group(3)
    basename = uri.rsplit("/", 1)[-1]
    return f'{head}file://{prefix}/{basename}{tail}'

new_src, n = mesh_re.subn(rewrite, src)

with open(path, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"    rewrote {n} <mesh> filename(s)")
PY

if (( CHECK )); then
    cat "${TMP_URDF}"
else
    install -m 0644 "${TMP_URDF}" "${OUT}"
    echo "==> wrote ${OUT}"
fi
