#!/usr/bin/env bash
#
# Installs (or upgrades) the latest bebop-agent + bebop-linux on the Jetson
# you're currently shelled into.
#
# bebop-linux is shipped as a single tarball (binary + bebop_v2.yaml +
# policy.onnx + policy.onnx.data + systemd unit) so the runtime, the
# joint config, and the policy weights all move together. The default
# source is the latest GitHub Release tagged `firmware/v*`. For
# pre-release main builds use `--run-id` to pull from a CI workflow run.
#
# bebop-agent still ships as a bare binary; its config / unit are fetched
# from `main` via the GitHub contents API as before.
#
# Usage:
#   sudo ./install-jetson.sh                  # latest firmware Release + latest green main agent
#   sudo ./install-jetson.sh --release firmware/v0.2.0   # pin bebop-linux to a tagged release
#   sudo ./install-jetson.sh --run-id 1234    # pin both daemons to a CI run (pre-release path)
#   sudo ./install-jetson.sh --branch dev     # latest green agent build on a branch
#   sudo ./install-jetson.sh --skip-prereqs   # don't touch system packages
#   sudo ./install-jetson.sh --agent-only     # skip bebop-linux
#   sudo ./install-jetson.sh --linux-only     # skip bebop-agent
#   sudo ./install-jetson.sh --setup-can      # also configure CAN: blacklist
#                                             # mttcan, load gs_usb, bring
#                                             # can* up at 1 Mbps via networkd
#   sudo ./install-jetson.sh --setup-can-only # just configure CAN; don't
#                                             # download or install binaries
#   sudo ./install-jetson.sh --build-gs-usb   # build gs_usb out-of-tree if
#                                             # the running kernel lacks it
#                                             # (JetPack stock kernel does);
#                                             # implies --setup-can
#   sudo ./install-jetson.sh --setup-imu      # also configure IMU access:
#                                             # udev rule giving the `bebop`
#                                             # group rw on /dev/spidev* and
#                                             # /dev/gpiochip* (so bebop-linux
#                                             # can open SPI + INT/RST GPIOs
#                                             # without root)
#   sudo ./install-jetson.sh --setup-imu-only # just configure IMU access;
#                                             # don't download or install
#                                             # binaries
#
# Requires:
#   * `gh` CLI authenticated (`gh auth login`) — needed to list/download
#     Releases, workflow artifacts, and (for the agent) the deploy/config
#     files via the contents API (works for private repos).
#   * arm64 Linux — the artifacts are aarch64 only.
#
# Idempotency:
#   * /etc/bebop/agent.toml is preserved if present (one-time bootstrap).
#   * /etc/bebop/bebop_v2.yaml AND /etc/bebop/policy.onnx{,.data} are
#     replaced unconditionally on every run — the firmware bundle is the
#     source of truth and they're versioned together as a unit. The
#     previous bebop_v2.yaml is saved as bebop_v2.yaml.bak so a bad
#     config push can be rolled back without re-downloading.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / args
# ---------------------------------------------------------------------------

REPO="${BEBOP_REPO:-ballerabdude/bebop}"
BRANCH="${BEBOP_BRANCH:-main}"
WORKFLOW="${BEBOP_WORKFLOW:-ci}"
# Glob matched against tag names when --release is not specified.
# Latest matching Release wins. Tag your firmware cuts as e.g.
# `firmware/v0.2.0` and they'll be picked up automatically.
RELEASE_GLOB="${BEBOP_RELEASE_GLOB:-firmware/v*}"

RUN_ID=""
RELEASE_TAG=""
SKIP_PREREQS=0
INSTALL_AGENT=1
INSTALL_LINUX=1
SETUP_CAN=0
SETUP_CAN_ONLY=0
BUILD_GS_USB=0
SETUP_IMU=0
SETUP_IMU_ONLY=0
# 1 Mbps is the Robstride bus rate; bebop-linux assumes the same.
CAN_BITRATE="${CAN_BITRATE:-1000000}"
# Group that owns /dev/spidev* and /dev/gpiochip* after `--setup-imu`.
# Defaults to `bebop` (which the JetPack OEM setup creates alongside the
# `bebop` login user); override if you run the runtime under a
# different account.
IMU_GROUP="${IMU_GROUP:-bebop}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    # Print every leading `#`-comment line up to the first blank line that
    # follows a comment block — i.e. the full doc header at the top of
    # this file, without us having to maintain a hard-coded line range.
    awk '
        NR == 1 { next }                       # skip shebang
        /^#/    { sub(/^# ?/, ""); print; next }
        /^$/    { exit }                       # stop at the first blank line
    ' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --run-id)         RUN_ID="$2"; shift 2 ;;
        --release)        RELEASE_TAG="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --workflow)       WORKFLOW="$2"; shift 2 ;;
        --repo)           REPO="$2"; shift 2 ;;
        --skip-prereqs)   SKIP_PREREQS=1; shift ;;
        --agent-only)     INSTALL_LINUX=0; shift ;;
        --linux-only)     INSTALL_AGENT=0; shift ;;
        --setup-can)      SETUP_CAN=1; shift ;;
        --setup-can-only) SETUP_CAN=1; SETUP_CAN_ONLY=1; shift ;;
        --build-gs-usb)   SETUP_CAN=1; BUILD_GS_USB=1; shift ;;
        --setup-imu)      SETUP_IMU=1; shift ;;
        --setup-imu-only) SETUP_IMU=1; SETUP_IMU_ONLY=1; shift ;;
        *)                echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ "${INSTALL_AGENT}" -eq 0 && "${INSTALL_LINUX}" -eq 0 ]]; then
    echo "--agent-only and --linux-only are mutually exclusive" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# CAN setup helper
# ---------------------------------------------------------------------------
#
# The Bebop V2 wiring uses a 5-port candleLight-style USB-CAN hub
# (Geschwister Schneider, USB id 1d50:606f, gs_usb driver). bebop_v2.yaml
# wants the buses at can0 / can1 / can4. The Orin Nano's *native* CAN0
# (mttcan, exposed on the 40-pin header) takes the can0 slot by default,
# so the USB hub gets bumped to can1..can5 and nothing matches.
#
# This function:
#   1. blacklists `mttcan` so the native controller never registers a netdev,
#      freeing can0 for the USB hub
#   2. ensures `gs_usb` (the USB-CAN driver) is loaded now and on every boot
#   3. drops a systemd-networkd .network drop-in that brings every can*
#      interface up at 1 Mbps automatically
#
# A reboot is the cleanest way to fully apply step 1 — `rmmod mttcan` may
# fail if anything has the device open. The function tries it best-effort
# and prints a clear "REBOOT REQUIRED" line when it can't.
setup_can() {
    echo "==> configuring CAN (gs_usb hub on Jetson Orin Nano)"

    # 1) Blacklist mttcan so the native CAN0 doesn't grab the can0 slot.
    install -d -m 0755 /etc/modprobe.d
    cat > /etc/modprobe.d/bebop-blacklist-mttcan.conf <<'EOF'
# Bebop: don't auto-load the Jetson Orin Nano native CAN driver. The
# robot uses an external USB-CAN hub (gs_usb / candleLight) and we want
# can0..canN to come from that hub, not from the SoC's mttcan
# controller. Remove this file (and reboot) to re-enable native CAN.
blacklist mttcan
EOF
    echo "    wrote /etc/modprobe.d/bebop-blacklist-mttcan.conf"

    # 2) Persist gs_usb on every boot, plus load it right now.
    install -d -m 0755 /etc/modules-load.d
    cat > /etc/modules-load.d/bebop-gs-usb.conf <<'EOF'
# Bebop: USB-CAN driver for the candleLight / Geschwister Schneider hub.
gs_usb
EOF
    echo "    wrote /etc/modules-load.d/bebop-gs-usb.conf"

    if modinfo gs_usb >/dev/null 2>&1; then
        modprobe gs_usb 2>/dev/null || true
    elif [[ "${BUILD_GS_USB}" -eq 1 ]]; then
        # JetPack's stock kernel ships without CONFIG_CAN_GS_USB. Build
        # the module out-of-tree against the running kernel's headers.
        echo "    gs_usb missing; building out-of-tree (--build-gs-usb)"
        if [[ -x "${SCRIPT_DIR}/build-gs-usb.sh" ]]; then
            "${SCRIPT_DIR}/build-gs-usb.sh"
        else
            echo "    ERROR: ${SCRIPT_DIR}/build-gs-usb.sh missing or not executable" >&2
            exit 1
        fi
    else
        cat >&2 <<EOF
    WARN: gs_usb not found in /lib/modules/$(uname -r). NVIDIA's stock
          JetPack 6 kernel ships without CONFIG_CAN_GS_USB, so the
          USB-CAN hub has no driver. Build the module out-of-tree:

              sudo ${SCRIPT_DIR}/build-gs-usb.sh

          ...or re-run this installer with --build-gs-usb.
EOF
    fi

    # 3) Bring all can* interfaces up at 1 Mbps via systemd-networkd.
    install -d -m 0755 /etc/systemd/network
    cat > /etc/systemd/network/80-bebop-can.network <<EOF
# Bebop: configure every can* netdev (i.e. every channel exposed by the
# USB-CAN hub) at the Robstride bus rate. systemd-networkd applies this
# whenever a matching interface appears, so plug-and-play works on next
# boot or hotplug.
[Match]
Name=can*

[CAN]
BitRate=${CAN_BITRATE}

[Link]
RequiredForOnline=no
EOF
    echo "    wrote /etc/systemd/network/80-bebop-can.network (bitrate=${CAN_BITRATE})"

    systemctl enable --now systemd-networkd >/dev/null 2>&1 \
        || echo "    WARN: failed to enable systemd-networkd" >&2

    # Best-effort: try to evict mttcan immediately so the USB hub claims
    # can0 without a reboot. This is fine if no one has the device open.
    local need_reboot=0
    if lsmod | awk '{print $1}' | grep -qx mttcan; then
        if rmmod mttcan 2>/dev/null; then
            echo "    rmmod mttcan succeeded; native can0 is gone"
        else
            echo "    NOTE: mttcan is still loaded (device busy); reboot to finish."
            need_reboot=1
        fi
    fi

    # Re-trigger systemd-networkd on whatever's already attached.
    systemctl restart systemd-networkd >/dev/null 2>&1 || true

    echo
    echo "    Current CAN interfaces:"
    ip -brief link show type can | sed 's/^/      /' || true

    if [[ "${need_reboot}" -eq 1 ]]; then
        echo
        echo "    REBOOT REQUIRED to fully drop the native mttcan controller."
        echo "    Run: sudo reboot"
    fi
}

# ---------------------------------------------------------------------------
# IMU setup helper
# ---------------------------------------------------------------------------
#
# The Bebop V2 IMU is a BNO085 wired for SPI (see
# `firmware/bebop-linux/config/bebop_v2.yaml` for the pinout). At
# runtime `bebop-linux` opens three device nodes:
#
#   * /dev/spidev0.0 (the SPI controller exposed by jetson-io's `spi1`)
#   * /dev/gpiochip0 (twice: line 144 for INT/HINTN, line 106 for RST)
#
# JetPack ships these as root-only (mode 0600, owner root:root), so the
# runtime fails its first `BNO08x::new_spi(...)` call with
# `PermissionDenied`. This function drops a udev rule that hands them
# to `${IMU_GROUP}` (default `bebop`, matching the OEM login group), so
# the runtime can come up as a regular service user without sudo.
#
# Caveat: enabling `spi1` itself is a one-time, *interactive*
# device-tree change made via `sudo /opt/nvidia/jetson-io/jetson-io.py`
# and requires a reboot. This function detects the missing
# `/dev/spidev0.0` and prints clear instructions instead of trying to
# automate that step (the jetson-io API is brittle enough that we'd
# rather a human run it once than fail half a setup mid-script).
setup_imu() {
    echo "==> configuring IMU access (SPI + GPIO udev rule, group=${IMU_GROUP})"

    # 1) Make sure the target group exists. JetPack OEM setup creates a
    #    `bebop` group alongside the `bebop` user; if someone's overriding
    #    IMU_GROUP and we can't find it, bail out clearly rather than
    #    silently writing a rule no user will benefit from.
    if ! getent group "${IMU_GROUP}" >/dev/null 2>&1; then
        cat >&2 <<EOF
    ERROR: group '${IMU_GROUP}' does not exist on this system. Either:

        # use a group that already exists (e.g. the JetPack default
        # 'bebop' if you're logged in as bebop, or just dialout):
        sudo IMU_GROUP=dialout $0 --setup-imu-only

        # or create one and add yourself to it:
        sudo groupadd ${IMU_GROUP}
        sudo usermod -aG ${IMU_GROUP} <your-login-user>
        # log out + back in, then re-run this script.
EOF
        exit 1
    fi

    # 2) Drop a udev rule covering every SPI controller and gpiochip
    #    on the system. We're not specific about which spidev / chip
    #    because the YAML config picks the active one — and bebop-linux
    #    refuses to start if it picks the wrong one. The rule is cheap
    #    (matches happen on device-add, no runtime cost).
    install -d -m 0755 /etc/udev/rules.d
    cat > /etc/udev/rules.d/99-bebop-imu.rules <<EOF
# Bebop V2 IMU (BNO085 over SPI + INT/RST GPIO).
#
# Hand /dev/spidev* and /dev/gpiochip* to the ${IMU_GROUP} group so the
# bebop-linux runtime can open the SPI bus and toggle the INT/RST
# GPIOs without root. Specific lines are picked up by gpiod inside the
# binary; see firmware/bebop-linux/config/bebop_v2.yaml for the active
# pinout. Remove this file to revert to the default root-only access.
KERNEL=="spidev*",   GROUP="${IMU_GROUP}", MODE="0660"
KERNEL=="gpiochip*", GROUP="${IMU_GROUP}", MODE="0660"
EOF
    echo "    wrote /etc/udev/rules.d/99-bebop-imu.rules"

    # 3) Reload + apply to nodes that are already present. The
    #    SUBSYSTEM matchers cover both the in-tree (`spidev`) and the
    #    legacy ("gpio") sysfs paths for the gpiochip devices.
    udevadm control --reload-rules
    udevadm trigger --subsystem-match=spidev 2>/dev/null || true
    udevadm trigger --subsystem-match=gpio   2>/dev/null || true

    # 4) Status: list whatever's there now so the operator can tell at a
    #    glance whether the rule actually took effect.
    echo
    echo "    Current IMU device nodes:"
    if compgen -G "/dev/spidev*" >/dev/null; then
        ls -l /dev/spidev* 2>/dev/null | sed 's/^/      /'
    else
        cat <<'EOF'
      (none — /dev/spidev0.0 is not present)
      The Jetson's SPI controller isn't enabled at the device-tree level.
      Run jetson-io to turn on `spi1` (40-pin header pins 19/21/23/24)
      and then reboot:

          sudo /opt/nvidia/jetson-io/jetson-io.py
          # → Configure 40-pin expansion header → Configure header pins
          #   manually → toggle `spi1` → Back → Save → Save and reboot
          sudo reboot

      After the reboot re-run `--setup-imu` (or the full installer) so
      this script can verify /dev/spidev0.0 came up.
EOF
    fi
    if compgen -G "/dev/gpiochip*" >/dev/null; then
        ls -l /dev/gpiochip* 2>/dev/null | sed 's/^/      /'
    else
        echo "      (none — no /dev/gpiochip* nodes found; very unusual on Jetson)"
    fi

    # 5) If the invoking user isn't already in the group, nudge them.
    #    Tilde-expanding $SUDO_USER on the way in works even when this
    #    script is run via `sudo -E` from a remote machine.
    if [[ -n "${SUDO_USER:-}" ]]; then
        if id -nG "${SUDO_USER}" 2>/dev/null | tr ' ' '\n' | grep -qx "${IMU_GROUP}"; then
            echo "    user '${SUDO_USER}' is already a member of '${IMU_GROUP}'"
        else
            echo
            echo "    NOTE: user '${SUDO_USER}' is NOT in group '${IMU_GROUP}'."
            echo "    Add them and log out + back in for the new group to take effect:"
            echo "        sudo usermod -aG ${IMU_GROUP} ${SUDO_USER}"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if [[ "${EUID}" -ne 0 ]]; then
    echo "install-jetson.sh must be run as root (sudo)" >&2
    exit 1
fi

# --setup-can-only / --setup-imu-only short-circuit before we touch
# gh / artifacts so an operator can run them from a freshly-cloned
# checkout without needing a CI build artifact to be available.
if [[ "${SETUP_CAN_ONLY}" -eq 1 || "${SETUP_IMU_ONLY}" -eq 1 ]]; then
    [[ "${SETUP_CAN_ONLY}" -eq 1 ]] && setup_can
    [[ "${SETUP_IMU_ONLY}" -eq 1 ]] && setup_imu
    exit 0
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    echo "WARN: host arch is ${ARCH}; CI publishes aarch64 binaries only." >&2
    echo "      Continuing anyway — this will almost certainly fail to run." >&2
fi

if ! command -v gh >/dev/null 2>&1; then
    cat >&2 <<'EOF'
gh CLI not found. Install it first, e.g.:

  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update && sudo apt-get install -y gh
  gh auth login
EOF
    exit 1
fi

# `gh auth status` is the canonical "am I logged in" probe. Under sudo we
# run as root, but most people run `gh auth login` from their normal
# user account — so root's credential store is empty even when the
# invoking user is logged in. If the calling user (SUDO_USER) is
# authenticated, lift their token into GH_TOKEN; gh honours that env
# var ahead of the on-disk credential store, and the rest of the
# script then "just works" without us having to wrap every call.
if ! gh auth status >/dev/null 2>&1; then
    if [[ -n "${SUDO_USER:-}" ]] \
        && sudo -u "${SUDO_USER}" -H gh auth status >/dev/null 2>&1; then
        echo "==> reusing gh auth from invoking user '${SUDO_USER}'"
        GH_TOKEN_FROM_USER="$(sudo -u "${SUDO_USER}" -H gh auth token 2>/dev/null || true)"
        if [[ -z "${GH_TOKEN_FROM_USER}" ]]; then
            echo "could not extract a gh token from ${SUDO_USER}; run 'sudo gh auth login' instead." >&2
            exit 1
        fi
        export GH_TOKEN="${GH_TOKEN_FROM_USER}"
    else
        cat >&2 <<EOF
gh is installed but not authenticated for the current user (root).

If you already ran 'gh auth login' as your normal user, you almost
certainly want one of:

  # easiest — re-run with the script (it will reuse SUDO_USER's auth):
  sudo $0 $*

  # or authenticate root explicitly:
  sudo gh auth login
EOF
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Resolve sources
# ---------------------------------------------------------------------------
#
# bebop-agent → CI workflow artifact (no Releases for the agent yet).
# bebop-linux → GitHub Release tagged `firmware/v*`, unless `--run-id`
#               is passed, in which case it falls back to the CI artifact
#               for that run (used for pre-release main builds).
#
# We only resolve the inputs we actually need: e.g. installing
# `--linux-only` from a Release shouldn't have to find a green CI run.

NEED_RUN_ID=0
NEED_RELEASE=0
if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
    NEED_RUN_ID=1
fi
if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
    if [[ -n "${RUN_ID}" ]]; then
        NEED_RUN_ID=1
    else
        NEED_RELEASE=1
    fi
fi

if [[ "${NEED_RUN_ID}" -eq 1 && -z "${RUN_ID}" ]]; then
    echo "==> resolving latest successful '${WORKFLOW}' run on ${REPO}@${BRANCH}"
    RUN_ID="$(gh run list \
        --repo "${REPO}" \
        --workflow "${WORKFLOW}" \
        --branch "${BRANCH}" \
        --status success \
        --limit 1 \
        --json databaseId \
        --jq '.[0].databaseId // empty')"
    if [[ -z "${RUN_ID}" ]]; then
        echo "no successful '${WORKFLOW}' run found on ${BRANCH}" >&2
        exit 1
    fi
fi
[[ -n "${RUN_ID}" ]] && echo "    using run id: ${RUN_ID}"

if [[ "${NEED_RELEASE}" -eq 1 && -z "${RELEASE_TAG}" ]]; then
    echo "==> resolving latest GitHub Release matching '${RELEASE_GLOB}'"
    # gh's `--exclude-pre-releases` / `--exclude-drafts` give us
    # production cuts only. The list is creation-time descending so
    # `.[0]` is the newest matching tag.
    RELEASE_TAG="$(gh release list \
        --repo "${REPO}" \
        --exclude-pre-releases \
        --exclude-drafts \
        --limit 50 \
        --json tagName \
        --jq "[.[] | select(.tagName | test(\"^${RELEASE_GLOB//\*/.*}$\"))] | .[0].tagName // empty")"
    if [[ -z "${RELEASE_TAG}" ]]; then
        cat >&2 <<EOF
no Release matching '${RELEASE_GLOB}' found on ${REPO}.

Either:
  * cut one by tagging a commit and pushing it:
      git tag firmware/v0.1.0 && git push origin firmware/v0.1.0
    (the 'ci' workflow's firmware-jetson job will build + publish it), or
  * install a pre-release build straight off main:
      sudo $0 --run-id <ci-run-id>
EOF
        exit 1
    fi
fi
[[ -n "${RELEASE_TAG}" ]] && echo "    using firmware release: ${RELEASE_TAG}"

# ---------------------------------------------------------------------------
# Stage everything in a tempdir so a partial failure leaves the system alone.
# ---------------------------------------------------------------------------

WORK_DIR="$(mktemp -d -t bebop-install.XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT
echo "==> staging in ${WORK_DIR}"

fetch_repo_file() {
    # Pull a file at HEAD of $BRANCH via the contents API (auth'd, works for
    # private repos and avoids the raw.githubusercontent CDN cache lag).
    local src="$1"
    local dst="$2"
    gh api \
        --header "Accept: application/vnd.github.raw" \
        "repos/${REPO}/contents/${src}?ref=${BRANCH}" \
        > "${dst}"
}

if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
    echo "==> downloading bebop-agent-aarch64 artifact"
    mkdir -p "${WORK_DIR}/agent-artifact"
    gh run download "${RUN_ID}" \
        --repo "${REPO}" \
        --name bebop-agent-aarch64 \
        --dir "${WORK_DIR}/agent-artifact"
    AGENT_BIN="${WORK_DIR}/agent-artifact/bebop-agent"
    if [[ ! -f "${AGENT_BIN}" ]]; then
        echo "bebop-agent binary missing from artifact" >&2
        exit 1
    fi
    chmod +x "${AGENT_BIN}"

    echo "==> fetching bebop-agent deploy assets"
    fetch_repo_file "jetson-agent/deploy/systemd/bebop-agent.service" \
        "${WORK_DIR}/bebop-agent.service"
    fetch_repo_file "jetson-agent/deploy/examples/agent.toml" \
        "${WORK_DIR}/agent.toml"
fi

if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
    # Download the firmware bundle (binary + bebop_v2.yaml + policy.onnx +
    # policy.onnx.data + bebop-linux.service + VERSION). The CI artifact
    # and Release asset are byte-identical and use the same filename.
    mkdir -p "${WORK_DIR}/linux-bundle" "${WORK_DIR}/linux-download"
    if [[ -n "${RELEASE_TAG}" ]]; then
        echo "==> downloading firmware bundle from release ${RELEASE_TAG}"
        gh release download "${RELEASE_TAG}" \
            --repo "${REPO}" \
            --pattern "bebop-linux-aarch64.tar.gz" \
            --pattern "bebop-linux-aarch64.tar.gz.sha256" \
            --clobber \
            --dir "${WORK_DIR}/linux-download"
    else
        echo "==> downloading firmware bundle from CI run ${RUN_ID}"
        gh run download "${RUN_ID}" \
            --repo "${REPO}" \
            --name bebop-linux-aarch64 \
            --dir "${WORK_DIR}/linux-download"
    fi

    LINUX_TARBALL="${WORK_DIR}/linux-download/bebop-linux-aarch64.tar.gz"
    if [[ ! -f "${LINUX_TARBALL}" ]]; then
        echo "bebop-linux bundle missing from download (expected ${LINUX_TARBALL})" >&2
        exit 1
    fi

    # Verify checksum when one was shipped. The CI step writes it next to
    # the tarball; older artifacts won't have one — don't hard-fail in
    # that case so emergency rollbacks to older Releases still work.
    if [[ -f "${WORK_DIR}/linux-download/bebop-linux-aarch64.tar.gz.sha256" ]]; then
        echo "==> verifying bundle checksum"
        (cd "${WORK_DIR}/linux-download" && sha256sum -c bebop-linux-aarch64.tar.gz.sha256)
    else
        echo "    (no .sha256 alongside the tarball — skipping checksum verify)"
    fi

    echo "==> extracting firmware bundle"
    tar -C "${WORK_DIR}/linux-bundle" -xzf "${LINUX_TARBALL}"

    LINUX_BIN="${WORK_DIR}/linux-bundle/bin/bebop-linux"
    LINUX_YAML="${WORK_DIR}/linux-bundle/config/bebop_v2.yaml"
    LINUX_ONNX="${WORK_DIR}/linux-bundle/config/policy.onnx"
    LINUX_ONNX_DATA="${WORK_DIR}/linux-bundle/config/policy.onnx.data"
    LINUX_UNIT="${WORK_DIR}/linux-bundle/systemd/bebop-linux.service"
    for f in "${LINUX_BIN}" "${LINUX_YAML}" "${LINUX_ONNX}" "${LINUX_ONNX_DATA}" "${LINUX_UNIT}"; do
        if [[ ! -f "${f}" ]]; then
            echo "bundle is missing $(basename "${f}") (looked at ${f})" >&2
            exit 1
        fi
    done
    chmod +x "${LINUX_BIN}"

    if [[ -f "${WORK_DIR}/linux-bundle/VERSION" ]]; then
        echo "    bundle VERSION:"
        sed 's/^/      /' "${WORK_DIR}/linux-bundle/VERSION"
    fi
fi

# ---------------------------------------------------------------------------
# Prereqs (only what bebop-agent strictly needs; bebop-linux is pure-Rust
# against SocketCAN and doesn't add anything new at install time).
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
    if systemctl list-unit-files "${unit}" 2>/dev/null | grep -q "${unit}"; then
        systemctl enable --now "${unit}" >/dev/null 2>&1 || true
    fi
}

if [[ "${SKIP_PREREQS}" -eq 0 && "${INSTALL_AGENT}" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "==> ensuring system prereqs (bluez, network-manager, dbus, docker)"
        apt_install_if_missing bluez network-manager dbus
        if ! command -v docker >/dev/null 2>&1; then
            apt_install_if_missing docker.io
        else
            echo "    already installed: docker ($(docker --version 2>/dev/null || echo unknown))"
        fi
        enable_unit_if_present bluetooth.service
        enable_unit_if_present NetworkManager.service
        enable_unit_if_present docker.service

        if ! command -v nvidia-ctk >/dev/null 2>&1 \
            && ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
            cat >&2 <<'EOF'

WARN: nvidia-container-toolkit not detected.
      The agent can still start, but the robot-app container will
      not get GPU access until you install it. On JetPack:
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker

EOF
        fi
    else
        echo "==> non-Debian system; skipping prereq install"
    fi
else
    echo "==> skipping prereq install"
fi

# ---------------------------------------------------------------------------
# Lay down files
# ---------------------------------------------------------------------------

install -d -m 0755 /etc/bebop /var/lib/bebop

if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
    echo "==> installing bebop-agent → /usr/local/bin/bebop-agent"
    install -m 0755 "${AGENT_BIN}" /usr/local/bin/bebop-agent

    if [[ ! -f /etc/bebop/agent.toml ]]; then
        echo "==> writing default /etc/bebop/agent.toml"
        install -m 0644 "${WORK_DIR}/agent.toml" /etc/bebop/agent.toml
    else
        echo "==> /etc/bebop/agent.toml already present, leaving as-is"
    fi

    install -m 0644 "${WORK_DIR}/bebop-agent.service" \
        /etc/systemd/system/bebop-agent.service
fi

if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
    echo "==> installing bebop-linux → /usr/local/bin/bebop-linux"
    install -m 0755 "${LINUX_BIN}" /usr/local/bin/bebop-linux

    # Replace bebop_v2.yaml on every install; back up the previous copy
    # so a bad rollout can be reverted without re-downloading. The
    # firmware bundle is the source of truth: joint limits, IMU pinout,
    # CAN bus assignments must all match the policy that ships with it.
    if [[ -f /etc/bebop/bebop_v2.yaml ]] \
       && ! cmp -s /etc/bebop/bebop_v2.yaml "${LINUX_YAML}"; then
        echo "==> updating /etc/bebop/bebop_v2.yaml (previous saved as .bak)"
        install -m 0644 /etc/bebop/bebop_v2.yaml /etc/bebop/bebop_v2.yaml.bak
    else
        echo "==> writing /etc/bebop/bebop_v2.yaml"
    fi
    install -m 0644 "${LINUX_YAML}" /etc/bebop/bebop_v2.yaml

    # Policy weights. `bebop-linux` resolves `--policy` to
    # `<config_dir>/policy.onnx` by default, so dropping both files
    # alongside the YAML is all the runtime needs. Both files MUST come
    # from the same training export — `policy.onnx` references
    # `policy.onnx.data` by relative path inside the graph.
    echo "==> installing policy → /etc/bebop/policy.onnx{,.data}"
    install -m 0644 "${LINUX_ONNX}"      /etc/bebop/policy.onnx
    install -m 0644 "${LINUX_ONNX_DATA}" /etc/bebop/policy.onnx.data

    install -m 0644 "${LINUX_UNIT}" /etc/systemd/system/bebop-linux.service
fi

# ---------------------------------------------------------------------------
# Hardware (opt-in). Both run before we (re)start bebop-linux so the
# bus + the IMU device nodes are usable by the time the runtime tries
# to open them.
# ---------------------------------------------------------------------------

if [[ "${SETUP_CAN}" -eq 1 ]]; then
    setup_can
fi

if [[ "${SETUP_IMU}" -eq 1 ]]; then
    setup_imu
fi

# ---------------------------------------------------------------------------
# Reload + start
# ---------------------------------------------------------------------------

echo "==> reloading systemd"
systemctl daemon-reload

if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
    echo "==> enabling + (re)starting bebop-agent"
    systemctl enable --now bebop-agent.service
    # If it was already running, the new binary needs a kick.
    systemctl restart bebop-agent.service
fi

if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
    echo "==> enabling + (re)starting bebop-linux"
    systemctl enable --now bebop-linux.service
    systemctl restart bebop-linux.service
fi

echo
echo "Done. Status:"
[[ "${INSTALL_AGENT}" -eq 1 ]] && systemctl --no-pager --lines=0 status bebop-agent || true
[[ "${INSTALL_LINUX}" -eq 1 ]] && systemctl --no-pager --lines=0 status bebop-linux || true

echo
echo "Logs:"
[[ "${INSTALL_AGENT}" -eq 1 ]] && echo "  journalctl -u bebop-agent -f" || true
[[ "${INSTALL_LINUX}" -eq 1 ]] && echo "  journalctl -u bebop-linux -f" || true
