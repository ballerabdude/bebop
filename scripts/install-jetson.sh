#!/usr/bin/env bash
#
# Installs (or upgrades) the latest bebop-agent + bebop-linux on the Jetson
# you're currently shelled into.
#
# bebop-linux is shipped as a single tarball (binary + bebop_v2.yaml +
# policy.onnx + policy.onnx.data + systemd unit, plus the optional
# navseg.onnx{,.data} navigable-path model when present in the checkout)
# so the runtime, the joint config, and the policy weights all move
# together. The default source is the latest GitHub Release tagged
# `firmware/v*`. For pre-release main builds use `--run-id` to pull from
# a CI workflow run.
#
# bebop-agent still ships as a bare binary; its config / unit are fetched
# from `main` via the GitHub contents API as before.
#
# Usage:
#   sudo ./install-jetson.sh                  # latest firmware Release + latest green main agent
#   sudo ./install-jetson.sh --release firmware/v0.2.0   # pin bebop-linux to a tagged release
#   sudo ./install-jetson.sh --run-id 1234    # pin both daemons to a CI run (pre-release path)
#   sudo ./install-jetson.sh --branch dev     # latest green agent build on a branch
#   sudo ./install-jetson.sh --local          # install from the local repo
#                                             # checkout (no GitHub, no `gh`):
#                                             # uses pre-built binaries from
#                                             # each crate's target/release/
#                                             # and config/deploy assets from
#                                             # the working tree. Pair with
#                                             # --build to also compile.
#   sudo ./install-jetson.sh --local --build  # like --local, but also runs
#                                             # `cargo build --release` for
#                                             # whichever daemons are being
#                                             # installed first (as SUDO_USER
#                                             # so target/ ownership stays
#                                             # sane).
#   sudo ./install-jetson.sh --local --repo-root /path/to/bebop
#                                             # use a different checkout than
#                                             # the one containing this script
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
#                                             # udev rules giving the `bebop`
#                                             # group rw on /dev/spidev* and
#                                             # /dev/gpiochip* (SPI backend) AND
#                                             # a stable /dev/bebop-imu symlink
#                                             # for the Teensy `imu_bridge`
#                                             # serial backend (so bebop-linux
#                                             # can open whichever the YAML
#                                             # selects without root)
#   sudo ./install-jetson.sh --setup-imu-only # just configure IMU access;
#                                             # don't download or install
#                                             # binaries
#   sudo ./install-jetson.sh --config-yaml bebop_wheeled.yaml
#                                             # activate a different robot
#                                             # config from the bundle
#                                             # (default: bebop_v2.yaml).
#                                             # The choice is remembered in
#                                             # /etc/bebop/config-yaml-name
#                                             # so plain upgrade runs keep
#                                             # it; the systemd unit gets a
#                                             # drop-in pointing ExecStart
#                                             # at the chosen file.
#   sudo ./install-jetson.sh --setup-orbbec   # also install Orbbec Gemini
#                                             # 335Lg depth-camera support:
#                                             # udev rules + the pinned
#                                             # pyorbbecsdk2 Python bindings
#   sudo ./install-jetson.sh --setup-orbbec-only
#                                             # just the Orbbec setup; don't
#                                             # download or install binaries
#   sudo ./install-jetson.sh --setup-orbbec --orbbec-venv /path/to/venv
#                                             # install the Python bindings
#                                             # into this venv instead of
#                                             # the auto-detected one
#
# Requires:
#   * `gh` CLI authenticated (`gh auth login`) — needed to list/download
#     Releases, workflow artifacts, and (for the agent) the deploy/config
#     files via the contents API (works for private repos). Not needed
#     when --local is used.
#   * arm64 Linux — the artifacts are aarch64 only.
#
# Idempotency:
#   * /etc/bebop/agent.toml is preserved if present (one-time bootstrap).
#   * /etc/bebop/bebop_v2.yaml AND /etc/bebop/policy.onnx{,.data} are
#     replaced unconditionally on every run — the firmware bundle is the
#     source of truth and they're versioned together as a unit. The
#     previous bebop_v2.yaml is saved as bebop_v2.yaml.bak so a bad
#     config push can be rolled back without re-downloading.
#   * /etc/bebop/navseg.onnx{,.data} are replaced when the bundle
#     carries them (same versioning rule); when a `nav:` block is
#     enabled in the YAML but the model files are missing, the install
#     WARNs — the firmware would soft-fail nav at boot.
#
# GPU note (one-time, NOT handled by this script): the nav runner needs
# the onnxruntime CUDA execution provider, which is a *separate* set of
# libraries under /usr/local/lib (libonnxruntime.so.1.23.0 built with
# --use_cuda, plus libonnxruntime_providers_shared.so and
# libonnxruntime_providers_cuda.so). Without them nav silently falls
# back to the CPU EP. See bebop-vision/README.md "GPU note" for the
# on-device source build. The installer prints a health check when it
# installs a nav model.
#
# Orbbec note (--setup-orbbec): the Gemini 335Lg (USB 2bc5:080b) must be
# on a USB 3.0 port — behind USB 2.0 it enumerates at 480 Mbps and
# cannot sustain full-resolution depth+color (the setup prints a warning
# when it detects that). The udev rules make Orbbec devices 0666, so no
# group membership or sudo is needed at runtime, and a udevadm trigger
# applies them to the already-plugged camera without a replug or reboot.
# The Python bindings are the pinned `pyorbbecsdk2` wheel (override with
# ORBBEC_SDK_VERSION=x.y.z); it bundles the OrbbecSDK v2 shared libs, so
# nothing is compiled on the Jetson, and the manylinux aarch64 wheels
# cover both JetPack 6.x (python3.10) and 7.x (python3.12). The target
# venv is resolved as: --orbbec-venv flag → <checkout>/bebop-vision/.venv
# when present (bebop-vision's documented venv convention) → a fresh
# /opt/bebop/orbbec-venv. Like every other process on the robot: only
# one process can hold the camera at a time — stop bebop-vision or
# OrbbecViewer before opening it from the other.

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
# --setup-orbbec: Orbbec Gemini 335Lg support. pyorbbecsdk2 is pinned so
# re-runs are reproducible; override via ORBBEC_SDK_VERSION=x.y.z env.
ORBBEC_SDK_VERSION="${ORBBEC_SDK_VERSION:-2.1.2}"
# --orbbec-venv: explicit venv for the Python bindings. Empty = auto:
# <checkout>/bebop-vision/.venv when present, else /opt/bebop/orbbec-venv.
ORBBEC_VENV=""
# --local: install from a local checkout instead of GitHub. Pre-built
# release binaries are picked up from each crate's target/release/ and
# configs/units come from the working tree. No `gh` required.
LOCAL=0
# Default repo root is the checkout containing this script.
LOCAL_REPO_ROOT=""
# When set with --local, run `cargo build --release` for each daemon
# being installed before staging. Done as SUDO_USER so target/ stays
# owned by the invoking user.
BUILD=0
# 1 Mbps is the Robstride bus rate; bebop-linux assumes the same.
CAN_BITRATE="${CAN_BITRATE:-1000000}"
# Group that owns /dev/spidev* and /dev/gpiochip* after `--setup-imu`.
# Defaults to `bebop` (which the JetPack OEM setup creates alongside the
# `bebop` login user); override if you run the runtime under a
# different account.
IMU_GROUP="${IMU_GROUP:-bebop}"
# Robot config to activate. Resolved in this order:
#   1. --config-yaml <name>          (explicit, and persisted for next time)
#   2. /etc/bebop/config-yaml-name   (a previous explicit choice)
#   3. bebop_v2.yaml                 (bundle default / current behavior)
# Must name a *.yaml shipped in the firmware bundle's config/ dir.
CONFIG_YAML=""

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
        --local)          LOCAL=1; shift ;;
        --build)          BUILD=1; shift ;;
        --repo-root)      LOCAL_REPO_ROOT="$2"; shift 2 ;;
        --skip-prereqs)   SKIP_PREREQS=1; shift ;;
        --agent-only)     INSTALL_LINUX=0; shift ;;
        --linux-only)     INSTALL_AGENT=0; shift ;;
        --setup-can)      SETUP_CAN=1; shift ;;
        --setup-can-only) SETUP_CAN=1; SETUP_CAN_ONLY=1; shift ;;
        --build-gs-usb)   SETUP_CAN=1; BUILD_GS_USB=1; shift ;;
        --setup-imu)      SETUP_IMU=1; shift ;;
        --setup-imu-only) SETUP_IMU=1; SETUP_IMU_ONLY=1; shift ;;
        --setup-orbbec)      SETUP_ORBBEC=1; shift ;;
        --setup-orbbec-only) SETUP_ORBBEC=1; SETUP_ORBBEC_ONLY=1; shift ;;
        --orbbec-venv)    ORBBEC_VENV="$2"; shift 2 ;;
        --config-yaml)    CONFIG_YAML="$2"; shift 2 ;;
        *)                echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ "${INSTALL_AGENT}" -eq 0 && "${INSTALL_LINUX}" -eq 0 ]]; then
    echo "--agent-only and --linux-only are mutually exclusive" >&2
    exit 2
fi

# `--local` is mutually exclusive with the GitHub-source selectors —
# they're meaningless when we're not talking to GitHub. Bail loudly so
# operators don't think their pin took effect.
if [[ "${LOCAL}" -eq 1 ]]; then
    if [[ -n "${RUN_ID}" || -n "${RELEASE_TAG}" ]]; then
        echo "--local cannot be combined with --run-id or --release" >&2
        exit 2
    fi
fi
if [[ "${LOCAL}" -eq 0 && "${BUILD}" -eq 1 ]]; then
    echo "--build only makes sense together with --local" >&2
    exit 2
fi
if [[ -n "${LOCAL_REPO_ROOT}" && "${LOCAL}" -eq 0 ]]; then
    echo "--repo-root only makes sense together with --local" >&2
    exit 2
fi

# Default --local checkout: the repo containing this script.
if [[ "${LOCAL}" -eq 1 && -z "${LOCAL_REPO_ROOT}" ]]; then
    LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ "${LOCAL}" -eq 1 && ! -d "${LOCAL_REPO_ROOT}" ]]; then
    echo "--repo-root '${LOCAL_REPO_ROOT}' is not a directory" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Resolve the robot config to activate (see CONFIG_YAML above).
# ---------------------------------------------------------------------------

if [[ -z "${CONFIG_YAML}" && -f /etc/bebop/config-yaml-name ]]; then
    CONFIG_YAML="$(cat /etc/bebop/config-yaml-name)"
    echo "==> using remembered config choice: ${CONFIG_YAML}"
fi
CONFIG_YAML="${CONFIG_YAML:-bebop_v2.yaml}"
# Refuse obviously-wrong values early (a path traversal or an empty
# string); existence inside the bundle is checked after extraction.
if [[ ! "${CONFIG_YAML}" =~ ^[A-Za-z0-9._-]+\.yaml$ ]]; then
    echo "--config-yaml must be a plain filename like 'bebop_wheeled.yaml' (got '${CONFIG_YAML}')" >&2
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
# The Bebop V2 IMU is a BNO085. There are two supported wiring
# topologies, selected by `imu.source` in
# `firmware/bebop-linux/config/bebop_v2.yaml`:
#
#   * source: spi    — the BNO is wired to the Jetson's SPI bus. At
#                      runtime `bebop-linux` opens three device nodes:
#                        - /dev/spidev0.0 (SPI controller via jetson-io `spi1`)
#                        - /dev/gpiochip0 (line 144 = INT/HINTN, 106 = RST)
#   * source: serial — the BNO is wired to the Teensy, which runs the
#                      `imu_bridge` firmware and streams binary frames to
#                      the Jetson over USB serial. `bebop-linux` opens a
#                      tty (e.g. /dev/ttyACM0). The Teensy enumerates with
#                      USB_DUAL_SERIAL as 16c0:048b: interface 0 is the
#                      binary frame stream, interface 2 is the debug log.
#
# JetPack ships /dev/spidev* and /dev/gpiochip* as root-only (mode 0600,
# owner root:root); /dev/ttyACM* are usually group `dialout`. So a
# non-root runtime fails to open them. This function drops a udev rule
# that hands all of them to `${IMU_GROUP}` (default `bebop`, matching the
# OEM login group) so the runtime can come up as a regular service user
# without sudo. For the serial backend it also creates stable symlinks
# /dev/bebop-imu (binary stream) and /dev/bebop-imu-debug (log), so the
# YAML can point at a name that doesn't shift when other USB CDC devices
# enumerate ahead of the Teensy.
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
# Bebop V2 IMU (BNO085). Covers both wiring topologies; the active one
# is chosen by 'imu.source' in
# firmware/bebop-linux/config/bebop_v2.yaml. Remove this file to revert
# to the default access.
#
# --- source: spi -----------------------------------------------------
# Hand /dev/spidev* and /dev/gpiochip* to the ${IMU_GROUP} group so the
# bebop-linux runtime can open the SPI bus and toggle the INT/RST
# GPIOs without root. Specific lines are picked up by gpiod inside the
# binary; see the YAML for the active pinout.
KERNEL=="spidev*",   GROUP="${IMU_GROUP}", MODE="0660"
KERNEL=="gpiochip*", GROUP="${IMU_GROUP}", MODE="0660"
#
# --- source: serial (Teensy imu_bridge) ------------------------------
# The Teensy enumerates with USB_DUAL_SERIAL as 16c0:048b and presents
# two CDC-ACM interfaces. Interface 0 carries the binary IMU frames;
# interface 2 is the human-readable debug log. Give the ${IMU_GROUP}
# group access and create stable symlinks so the YAML 'serial_device'
# can point at /dev/bebop-imu regardless of ttyACM enumeration order.
SUBSYSTEM=="tty", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="048b", ENV{ID_USB_INTERFACE_NUM}=="00", GROUP="${IMU_GROUP}", MODE="0660", SYMLINK+="bebop-imu"
SUBSYSTEM=="tty", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="048b", ENV{ID_USB_INTERFACE_NUM}=="02", GROUP="${IMU_GROUP}", MODE="0660", SYMLINK+="bebop-imu-debug"
EOF
    echo "    wrote /etc/udev/rules.d/99-bebop-imu.rules"

    # 3) Reload + apply to nodes that are already present. The
    #    SUBSYSTEM matchers cover the in-tree (`spidev`), the legacy
    #    ("gpio") sysfs path for gpiochip devices, and the Teensy tty.
    udevadm control --reload-rules
    udevadm trigger --subsystem-match=spidev 2>/dev/null || true
    udevadm trigger --subsystem-match=gpio   2>/dev/null || true
    udevadm trigger --subsystem-match=tty    2>/dev/null || true
    # `trigger` only *queues* events; wait for the daemon to process them
    # so the status listing below reflects the symlinks it just created
    # (otherwise /dev/bebop-imu can race and show up as "not present").
    udevadm settle 2>/dev/null || true

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
    echo
    echo "    Teensy imu_bridge serial (source: serial):"
    if compgen -G "/dev/bebop-imu*" >/dev/null; then
        ls -l /dev/bebop-imu* 2>/dev/null | sed 's/^/      /'
    else
        cat <<'EOF'
      (none — /dev/bebop-imu not present)
      Either the Teensy isn't plugged in / flashed with the `imu_bridge`
      firmware, or it's running a non-dual-serial USB type. Flash it with:

          pio run -e imu_bridge --target upload   # from firmware/bebop-locomotion

      then re-run `--setup-imu`. Only needed when bebop_v2.yaml sets
      `imu.source: "serial"`; ignore this for the SPI backend.
EOF
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
# Orbbec depth camera setup helper
# ---------------------------------------------------------------------------
#
# bebop-vision is gaining an Orbbec Gemini 335Lg (USB 2bc5:080b) as a
# directly-opened depth sensor. It is a second, physically separate
# camera — the firmware's OBSBOT PTZ keeps its exclusive /dev/video*
# claim, so there is no ownership conflict with bebop-linux.
#
# A non-root Python process needs three things:
#
#   1. udev rules. JetPack ships USB device nodes root-only. The vendored
#      scripts/orbbec-99-obsensor-libusb.rules (Orbbec's official rules,
#      covering the whole 2bc5 product range) makes the camera 0666.
#   2. Python bindings. The `pyorbbecsdk2` wheel from PyPI bundles the
#      OrbbecSDK v2 shared libraries — nothing to build on the Jetson,
#      and the manylinux_2_27 aarch64 wheels span cp38–cp313, i.e. both
#      JetPack 6.x (Ubuntu 22.04, python3.10) and 7.x (24.04, python3.12)
#      system Pythons.
#   3. A venv that owns the install. bebop-vision's convention is a venv
#      in the checkout (bebop-vision/.venv, per its README bootstrap).
#      This function installs into that venv when it exists on the
#      robot; otherwise it creates a dedicated /opt/bebop/orbbec-venv so
#      the bindings are ready before the bebop-vision checkout lands.
#      --orbbec-venv overrides either choice.
#
# No reboot is required: `udevadm trigger` applies the rules to the
# already-enumerated device. A camera currently held open by another
# process keeps its old permissions until that process exits — unplug/
# replug is the sledgehammer if a trigger ever seems to not take.
setup_orbbec() {
    echo "==> configuring Orbbec Gemini 335Lg access"

    # 1) Resolve the target venv (see the precedence comment above).
    local venv=""
    if [[ -n "${ORBBEC_VENV}" ]]; then
        venv="${ORBBEC_VENV}"
    else
        local checkout_root="${LOCAL_REPO_ROOT}"
        if [[ -z "${checkout_root}" ]]; then
            checkout_root="$(cd "${SCRIPT_DIR}/.." && pwd)"
        fi
        if [[ -x "${checkout_root}/bebop-vision/.venv/bin/pip" ]]; then
            venv="${checkout_root}/bebop-vision/.venv"
            echo "    using existing bebop-vision venv: ${venv}"
        else
            venv="/opt/bebop/orbbec-venv"
            echo "    no bebop-vision venv found; creating ${venv}"
            install -d -m 0755 "$(dirname "${venv}")"
            # Stock JetPack images ship without python3-venv (the venv
            # module needs its ensurepip pieces). Install on demand and
            # retry once before giving up.
            if ! python3 -m venv "${venv}" 2>/dev/null; then
                if command -v apt-get >/dev/null 2>&1; then
                    echo "    venv creation failed; installing python3-venv"
                    DEBIAN_FRONTEND=noninteractive apt-get update -qq
                    DEBIAN_FRONTEND=noninteractive apt-get install -y \
                        --no-install-recommends python3-venv
                fi
                if ! python3 -m venv "${venv}"; then
                    cat >&2 <<EOF
    ERROR: could not create venv at ${venv}.

          python3 with the venv module (python3-venv on Debian/Ubuntu)
          is required, or point at an existing venv:

              sudo $0 --setup-orbbec-only --orbbec-venv /path/to/venv
EOF
                    exit 1
                fi
            fi
        fi
    fi
    if [[ ! -x "${venv}/bin/pip" ]]; then
        cat >&2 <<EOF
    ERROR: '${venv}' does not look like a usable venv (no bin/pip).

          Create it first:

              python3 -m venv ${venv} && ${venv}/bin/pip install --upgrade pip

          ...or pass a different one:

              sudo $0 --setup-orbbec-only --orbbec-venv /path/to/venv
EOF
        exit 1
    fi

    # 2) Python bindings. Pinned version (ORBBEC_SDK_VERSION) so upgrade
    #    runs are reproducible; bump deliberately when integrating.
    echo "==> installing pyorbbecsdk2==${ORBBEC_SDK_VERSION} into ${venv}"
    "${venv}/bin/pip" install --upgrade "pyorbbecsdk2==${ORBBEC_SDK_VERSION}"

    # 3) udev rules from the vendored copy. Installing over an existing
    #    file is the upgrade path (new PIDs ship with SDK releases).
    local rules_src="${SCRIPT_DIR}/orbbec-99-obsensor-libusb.rules"
    if [[ ! -f "${rules_src}" ]]; then
        echo "    ERROR: vendored udev rules missing: ${rules_src}" >&2
        exit 1
    fi
    install -d -m 0755 /etc/udev/rules.d
    install -m 0644 "${rules_src}" /etc/udev/rules.d/99-obsensor-libusb.rules
    echo "    wrote /etc/udev/rules.d/99-obsensor-libusb.rules"
    udevadm control --reload-rules
    udevadm trigger --subsystem-match=usb 2>/dev/null || udevadm trigger
    udevadm settle 2>/dev/null || true

    # 4) Verify. Absence of a camera is a WARN, not a failure — this
    #    often runs on a robot before the camera is mounted.
    echo
    echo "    Orbbec device check:"
    if ! "${venv}/bin/python" - <<'PYEOF'
import sys

import pyorbbecsdk as ob

# Keep the Context in a named variable: the DeviceList returned by
# query_devices() borrows the context's internal device manager, and a
# temporary `ob.Context().query_devices()` would destroy it on the same
# line ("NULL pointer passed for argument deviceMgr").
ctx = ob.Context()
devices = ctx.query_devices()
count = devices.get_count()
if count == 0:
    print("      (no Orbbec device found — plug it into a USB 3.0 port;")
    print("       the udev rules are in place, no re-run needed)")
    sys.exit(0)
for i in range(count):
    info = devices.get_device_by_index(i).get_device_info()
    print(f"      {info.get_name()}  serial={info.get_serial_number()}"
          f"  fw={info.get_firmware_version()}")
PYEOF
    then
        cat >&2 <<EOF
    WARN: pyorbbecsdk2 import/enumeration failed in ${venv}.
          The pip install above may have picked a wheel incompatible
          with this Python — check \`${venv}/bin/python --version\`.
EOF
    fi

    # 5) USB link-speed sanity from sysfs. The SDK can't report this;
    #    a 335Lg behind USB 2.0 will frustrate whoever debugs streaming
    #    later, so flag it now.
    local dev_dir usb_speed found=0
    for dev_dir in /sys/bus/usb/devices/*; do
        if [[ -f "${dev_dir}/idVendor" ]] && grep -qx 2bc5 "${dev_dir}/idVendor"; then
            found=1
            usb_speed="$(cat "${dev_dir}/speed" 2>/dev/null || echo unknown)"
            if [[ "${usb_speed}" =~ ^(1.5|12|480)$ ]]; then
                echo
                echo "    WARN: Orbbec device on ${dev_dir##*/} is linked at ${usb_speed} Mbps (USB 2.0)." >&2
                echo "          The Gemini 335Lg needs a USB 3.0 port for full resolution/fps;" >&2
                echo "          move it to a direct (non-hub) USB 3.0 port." >&2
            else
                echo "    USB link speed: ${usb_speed} Mbps (USB 3.0, OK)"
            fi
        fi
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "    (no Orbbec device on the bus right now)"
    fi

    echo
    echo "    Orbbec setup complete. Try it:"
    echo "        ${venv}/bin/python -c 'import pyorbbecsdk as ob; ctx = ob.Context(); print(ctx.query_devices().get_count(), \"device(s)\")'"
}

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if [[ "${EUID}" -ne 0 ]]; then
    echo "install-jetson.sh must be run as root (sudo)" >&2
    exit 1
fi

# --setup-can-only / --setup-imu-only / --setup-orbbec-only short-circuit
# before we touch gh / artifacts so an operator can run them from a
# freshly-cloned checkout without needing a CI build artifact to be
# available.
if [[ "${SETUP_CAN_ONLY}" -eq 1 || "${SETUP_IMU_ONLY}" -eq 1 || "${SETUP_ORBBEC_ONLY}" -eq 1 ]]; then
    [[ "${SETUP_CAN_ONLY}" -eq 1 ]] && setup_can
    [[ "${SETUP_IMU_ONLY}" -eq 1 ]] && setup_imu
    [[ "${SETUP_ORBBEC_ONLY}" -eq 1 ]] && setup_orbbec
    exit 0
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    echo "WARN: host arch is ${ARCH}; CI publishes aarch64 binaries only." >&2
    echo "      Continuing anyway — this will almost certainly fail to run." >&2
fi

if [[ "${LOCAL}" -eq 1 ]]; then
    echo "==> --local: installing from ${LOCAL_REPO_ROOT}"
fi

if [[ "${LOCAL}" -eq 0 ]] && ! command -v gh >/dev/null 2>&1; then
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
if [[ "${LOCAL}" -eq 0 ]] && ! gh auth status >/dev/null 2>&1; then
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
if [[ "${LOCAL}" -eq 0 ]]; then
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

# Run a command as the invoking user (SUDO_USER) when we're under sudo,
# preserving the user's PATH so cargo/rustup shims resolve. Falls back
# to running directly if there's no SUDO_USER (e.g. true root login).
run_as_user() {
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        sudo -u "${SUDO_USER}" -H bash -lc "$*"
    else
        bash -lc "$*"
    fi
}

# --build (only valid with --local): compile each daemon being installed
# before staging. Done as SUDO_USER so target/ keeps user-writable
# permissions across re-runs without sudo.
if [[ "${LOCAL}" -eq 1 && "${BUILD}" -eq 1 ]]; then
    if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
        echo "==> building bebop-agent (release) in ${LOCAL_REPO_ROOT}/jetson-agent"
        run_as_user "cd '${LOCAL_REPO_ROOT}/jetson-agent' && cargo build --release -p bebop-agent"
    fi
    if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
        echo "==> building bebop-linux (release) in ${LOCAL_REPO_ROOT}/firmware/bebop-linux"
        run_as_user "cd '${LOCAL_REPO_ROOT}/firmware/bebop-linux' && cargo build --release"
    fi
fi

if [[ "${INSTALL_AGENT}" -eq 1 ]]; then
    if [[ "${LOCAL}" -eq 1 ]]; then
        echo "==> staging bebop-agent from local checkout"
        AGENT_BIN_SRC="${LOCAL_REPO_ROOT}/jetson-agent/target/release/bebop-agent"
        AGENT_UNIT_SRC="${LOCAL_REPO_ROOT}/jetson-agent/deploy/systemd/bebop-agent.service"
        AGENT_TOML_SRC="${LOCAL_REPO_ROOT}/jetson-agent/deploy/examples/agent.toml"
        if [[ ! -f "${AGENT_BIN_SRC}" ]]; then
            cat >&2 <<EOF
bebop-agent binary not found at:
    ${AGENT_BIN_SRC}

Build it first (or re-run with --build):
    (cd ${LOCAL_REPO_ROOT}/jetson-agent && cargo build --release -p bebop-agent)
EOF
            exit 1
        fi
        for f in "${AGENT_UNIT_SRC}" "${AGENT_TOML_SRC}"; do
            if [[ ! -f "${f}" ]]; then
                echo "missing local deploy asset: ${f}" >&2
                exit 1
            fi
        done
        mkdir -p "${WORK_DIR}/agent-artifact"
        install -m 0755 "${AGENT_BIN_SRC}"  "${WORK_DIR}/agent-artifact/bebop-agent"
        install -m 0644 "${AGENT_UNIT_SRC}" "${WORK_DIR}/bebop-agent.service"
        install -m 0644 "${AGENT_TOML_SRC}" "${WORK_DIR}/agent.toml"
        AGENT_BIN="${WORK_DIR}/agent-artifact/bebop-agent"
    else
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
fi

if [[ "${INSTALL_LINUX}" -eq 1 ]]; then
    # Source layout (binary + bebop_v2.yaml + policy.onnx + policy.onnx.data +
    # bebop-linux.service + VERSION) matches what the CI bundle ships, so
    # everything downstream of "extract" is identical between modes.
    mkdir -p "${WORK_DIR}/linux-bundle" "${WORK_DIR}/linux-download"
    if [[ "${LOCAL}" -eq 1 ]]; then
        echo "==> staging bebop-linux bundle from local checkout"
        LX_ROOT="${LOCAL_REPO_ROOT}/firmware/bebop-linux"
        LX_BIN_SRC="${LX_ROOT}/target/release/bebop-linux"
        LX_ONNX_SRC="${LX_ROOT}/config/policy.onnx"
        LX_ONNX_DATA_SRC="${LX_ROOT}/config/policy.onnx.data"
        LX_UNIT_SRC="${LX_ROOT}/deploy/systemd/bebop-linux.service"
        if [[ ! -f "${LX_BIN_SRC}" ]]; then
            cat >&2 <<EOF
bebop-linux binary not found at:
    ${LX_BIN_SRC}

Build it first (or re-run with --build):
    (cd ${LX_ROOT} && cargo build --release)
EOF
            exit 1
        fi
        for f in "${LX_ONNX_SRC}" "${LX_ONNX_DATA_SRC}" "${LX_UNIT_SRC}"; do
            if [[ ! -f "${f}" ]]; then
                echo "missing local firmware asset: ${f}" >&2
                exit 1
            fi
        done
        install -d "${WORK_DIR}/linux-bundle/bin" \
                    "${WORK_DIR}/linux-bundle/config" \
                    "${WORK_DIR}/linux-bundle/systemd"
        install -m 0755 "${LX_BIN_SRC}"        "${WORK_DIR}/linux-bundle/bin/bebop-linux"
        # Every config variant, mirroring CI's bundle layout; the
        # active one is picked by CONFIG_YAML below.
        for f in "${LX_ROOT}"/config/*.yaml; do
            [[ -e "${f}" ]] || continue
            install -m 0644 "${f}" "${WORK_DIR}/linux-bundle/config/$(basename "${f}")"
        done
        install -m 0644 "${LX_ONNX_SRC}"       "${WORK_DIR}/linux-bundle/config/policy.onnx"
        install -m 0644 "${LX_ONNX_DATA_SRC}"  "${WORK_DIR}/linux-bundle/config/policy.onnx.data"
        # Optional nav model: stage when present so the local bundle
        # matches CI's (the runtime resolves <config_dir>/navseg.onnx
        # only when the YAML enables `nav:` — absent files are fine).
        for f in "${LX_ROOT}/config/navseg.onnx" "${LX_ROOT}/config/navseg.onnx.data"; do
            if [[ -f "${f}" ]]; then
                install -m 0644 "${f}" "${WORK_DIR}/linux-bundle/config/$(basename "${f}")"
            fi
        done
        install -m 0644 "${LX_UNIT_SRC}"       "${WORK_DIR}/linux-bundle/systemd/bebop-linux.service"
        # Synthesize a VERSION file mirroring CI's, so the post-install
        # echo gives operators something useful to grep in journals.
        local_sha="$(run_as_user "git -C '${LOCAL_REPO_ROOT}' rev-parse HEAD 2>/dev/null" || true)"
        local_dirty=""
        if [[ -n "${local_sha}" ]] \
            && ! run_as_user "git -C '${LOCAL_REPO_ROOT}' diff --quiet HEAD 2>/dev/null"; then
            local_dirty="-dirty"
        fi
        {
            echo "sha=${local_sha:-unknown}${local_dirty}"
            echo "ref=local"
            echo "repo_root=${LOCAL_REPO_ROOT}"
            echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "${WORK_DIR}/linux-bundle/VERSION"
    else
        # Download the firmware bundle. The CI artifact and Release asset
        # are byte-identical and use the same filename.
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

        # Verify checksum when one was shipped. The CI step writes it
        # next to the tarball; older artifacts won't have one — don't
        # hard-fail in that case so emergency rollbacks to older
        # Releases still work.
        if [[ -f "${WORK_DIR}/linux-download/bebop-linux-aarch64.tar.gz.sha256" ]]; then
            echo "==> verifying bundle checksum"
            (cd "${WORK_DIR}/linux-download" && sha256sum -c bebop-linux-aarch64.tar.gz.sha256)
        else
            echo "    (no .sha256 alongside the tarball — skipping checksum verify)"
        fi

        echo "==> extracting firmware bundle"
        tar -C "${WORK_DIR}/linux-bundle" -xzf "${LINUX_TARBALL}"
    fi

    LINUX_BIN="${WORK_DIR}/linux-bundle/bin/bebop-linux"
    # The active robot config, per CONFIG_YAML (flag / remembered /
    # default). The bundle carries every variant; we activate one.
    LINUX_YAML="${WORK_DIR}/linux-bundle/config/${CONFIG_YAML}"
    LINUX_ONNX="${WORK_DIR}/linux-bundle/config/policy.onnx"
    LINUX_ONNX_DATA="${WORK_DIR}/linux-bundle/config/policy.onnx.data"
    LINUX_UNIT="${WORK_DIR}/linux-bundle/systemd/bebop-linux.service"
    # Optional nav model — presence-checked below, never required (the
    # firmware soft-fails nav when the files are absent).
    LINUX_NAV_ONNX="${WORK_DIR}/linux-bundle/config/navseg.onnx"
    LINUX_NAV_ONNX_DATA="${WORK_DIR}/linux-bundle/config/navseg.onnx.data"
    for f in "${LINUX_BIN}" "${LINUX_YAML}" "${LINUX_ONNX}" "${LINUX_ONNX_DATA}" "${LINUX_UNIT}"; do
        if [[ ! -f "${f}" ]]; then
            echo "bundle is missing $(basename "${f}") (looked at ${f})" >&2
            if [[ "${f}" == "${LINUX_YAML}" ]]; then
                echo "the active config is '${CONFIG_YAML}'; available in this bundle:" >&2
                ls "${WORK_DIR}/linux-bundle/config/"*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^/    /' >&2
            fi
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

# Firmware prereqs. v4l-utils ships `v4l2-ctl`, the operator + bring-up
# surface for the camera (the OBSBOT Tiny 2's PTZ rides standard UVC
# pan/tilt controls on /dev/video0; the firmware talks raw ioctls, but
# humans use v4l2-ctl to inspect ranges and jog the gimbal by hand):
#   v4l2-ctl -d /dev/video0 --list-ctrls     # pan/tilt/zoom + ranges
#   v4l2-ctl -d /dev/video0 --get-ctrl pan_absolute
# Gated on INSTALL_LINUX like the agent prereqs above are gated on the
# agent, so --skip-prereqs and --agent-only keep their meaning.
if [[ "${SKIP_PREREQS}" -eq 0 && "${INSTALL_LINUX}" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "==> ensuring system prereqs (v4l-utils)"
        apt_install_if_missing v4l-utils
    else
        echo "==> non-Debian system; skipping v4l-utils"
    fi
fi

# Health check for the onnxruntime CUDA provider libraries the nav
# runner dlopens at runtime. The CUDA EP ships as THREE files (main lib
# built with --use_cuda + the provider bridge + the CUDA EP itself); a
# main-lib-only install accepts CUDA sessions but silently runs
# everything on the CPU EP. Warn loudly rather than fail — nav is
# best-effort and the CPU fallback still works.
nav_gpu_check() {
    local ort_dir
    for ort_dir in /usr/local/lib /usr/lib/aarch64-linux-gnu; do
        if [[ -f "${ort_dir}/libonnxruntime_providers_cuda.so" \
           && -f "${ort_dir}/libonnxruntime_providers_shared.so" ]]; then
            echo "    nav GPU: CUDA EP present (${ort_dir}/libonnxruntime_providers_cuda.so)"
            return 0
        fi
    done
    cat >&2 <<EOF
    WARN: onnxruntime CUDA provider libraries not found — the nav
          runner will fall back to the CPU EP (a few Hz instead of
          ~20 Hz on the Orin GPU). The CUDA build must match the
          firmware's ort crate (onnxruntime 1.23.0) and installs as
          three files under /usr/local/lib:

              libonnxruntime.so.1.23.0              (CUDA-enabled build)
              libonnxruntime_providers_shared.so    (provider bridge)
              libonnxruntime_providers_cuda.so      (the CUDA EP)

          Keep the CPU original as libonnxruntime.so.1.23.0.cpu-backup.
          Build recipe: bebop-vision/README.md, "GPU note".
EOF
}

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

    # Robot config. Installed under its REAL filename (e.g.
    # /etc/bebop/bebop_wheeled.yaml); the systemd unit's ExecStart is
    # repointed via a drop-in when it isn't the historical default
    # (/etc/bebop/bebop_v2.yaml, which the shipped unit references).
    # Backup logic applies to whichever file is active so a bad rollout
    # can be reverted without re-downloading.
    yaml_dst="/etc/bebop/${CONFIG_YAML}"
    if [[ -f "${yaml_dst}" ]] && ! cmp -s "${yaml_dst}" "${LINUX_YAML}"; then
        echo "==> updating ${yaml_dst} (previous saved as .bak)"
        install -m 0644 "${yaml_dst}" "${yaml_dst}.bak"
    else
        echo "==> writing ${yaml_dst}"
    fi
    install -m 0644 "${LINUX_YAML}" "${yaml_dst}"

    # Remember the choice so plain upgrade runs don't flip the robot
    # back to the default config. Written unconditionally (also pins
    # the default when the operator never passes the flag — harmless,
    # and makes an explicit later switch observable).
    echo "${CONFIG_YAML}" > /etc/bebop/config-yaml-name

    # Non-default config: drop-in overrides the unit's --config path.
    # A drop-in (not editing the unit) survives this installer
    # rewriting the base unit on every upgrade. Switching back to the
    # default removes the drop-in.
    unit_dropin_dir=/etc/systemd/system/bebop-linux.service.d
    if [[ "${CONFIG_YAML}" != "bebop_v2.yaml" ]]; then
        install -d -m 0755 "${unit_dropin_dir}"
        cat > "${unit_dropin_dir}/10-config.conf" <<EOF
# Managed by install-jetson.sh --config-yaml ${CONFIG_YAML}.
# Overrides ExecStart to activate this robot's config variant.
[Service]
ExecStart=
ExecStart=/usr/local/bin/bebop-linux \\
    --config ${yaml_dst} \\
    --capture-dir /var/lib/bebop-captures
EOF
        echo "==> systemd drop-in: ${unit_dropin_dir}/10-config.conf (--config ${yaml_dst})"
    else
        rm -f "${unit_dropin_dir}/10-config.conf"
    fi

    # Policy weights. `bebop-linux` resolves `--policy` to
    # `<config_dir>/policy.onnx` by default, so dropping both files
    # alongside the YAML is all the runtime needs. Both files MUST come
    # from the same training export — `policy.onnx` references
    # `policy.onnx.data` by relative path inside the graph.
    echo "==> installing policy → /etc/bebop/policy.onnx{,.data}"
    install -m 0644 "${LINUX_ONNX}"      /etc/bebop/policy.onnx
    install -m 0644 "${LINUX_ONNX_DATA}" /etc/bebop/policy.onnx.data

    # Navigable-path model (optional, same drop-in convention as the
    # policy). The runtime loads <config_dir>/navseg.onnx when the YAML
    # has a `nav:` block; both files must come from the same export.
    if [[ -f "${LINUX_NAV_ONNX}" && -f "${LINUX_NAV_ONNX_DATA}" ]]; then
        echo "==> installing nav model → /etc/bebop/navseg.onnx{,.data}"
        install -m 0644 "${LINUX_NAV_ONNX}"      /etc/bebop/navseg.onnx
        install -m 0644 "${LINUX_NAV_ONNX_DATA}" /etc/bebop/navseg.onnx.data
        nav_gpu_check
    elif grep -qE '^nav:' "${LINUX_YAML}" 2>/dev/null; then
        echo "    WARN: config enables \`nav:\` but navseg.onnx{,.data} are missing" >&2
        echo "          from the bundle — nav will soft-fail at boot." >&2
    fi

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

if [[ "${SETUP_ORBBEC}" -eq 1 ]]; then
    setup_orbbec
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
