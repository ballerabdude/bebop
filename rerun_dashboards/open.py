"""Open navd sessions in the Rerun viewer with the bebop_navd dashboard.

    python rerun_dashboards/open.py <session.mcap | URL> [...]

    python rerun_dashboards/open.py \\
        http://bebop.local:9090/captures/dl/navd_session_20261001_223516.mcap

Pass the firmware's always-on system log(s) as extra paths to populate the
`power` / `host` views (Protobuf `bebop.system.*`, `system_*.mcap`); they
rotate hourly, so a long drive may span more than one file:

    python rerun_dashboards/open.py navd_session_*.mcap system_*.mcap

A URL is downloaded once into the cache dir (default
`~/.cache/bebop/captures`, override with `--cache-dir`) and reused while
the server-side size matches.

Why the SDK does the loading: the viewer only applies a blueprint to
recordings with the same application id, and opening a `.mcap` directly
(`rerun x.mcap`) names the app after the file, so a saved blueprint never
matches. Loading via `log_file_from_path` in a stream initialised as
`bebop_navd` gives the data the dashboard's app id, and the blueprint is
sent straight from `navd_blueprint.build()` — no `.rbl` or conversion.

Needs the rerun SDK (e.g. `pipx install rerun-sdk` and run with that
venv's python, or `pip install rerun-sdk`); keep it on the same version as
the viewer.
"""

import argparse
import os
import shutil
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerun as rr  # noqa: E402

import navd_blueprint  # noqa: E402

DEFAULT_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "bebop", "captures")


def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=10) as resp:
        size = resp.headers.get("Content-Length")
    return int(size) if size is not None else None


def _fetch(url, cache_dir):
    """Download `url` into `cache_dir` (reusing a size-matched copy)."""
    name = os.path.basename(urllib.parse.urlparse(url).path)
    if not name:
        sys.exit(f"cannot derive a file name from {url}")
    dest = os.path.join(cache_dir, name)
    size = _remote_size(url)
    if os.path.exists(dest) and size is not None \
            and os.path.getsize(dest) == size:
        print(f"cached: {dest}")
        return dest

    os.makedirs(cache_dir, exist_ok=True)
    tmp = dest + ".part"
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=30) as resp, \
            open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done >> 20}/{total >> 20} MiB", end="",
                      flush=True)
    print()
    if total and done != total:
        os.remove(tmp)
        sys.exit(f"short download ({done}/{total} bytes): {url}")
    os.replace(tmp, dest)
    return dest


def _resolve(src, cache_dir):
    if urllib.parse.urlparse(src).scheme in ("http", "https"):
        return _fetch(src, cache_dir)
    if not os.path.isfile(src):
        sys.exit(f"no such file: {src}")
    return src


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+",
                    help="navd session .mcap path(s) or http(s) URL(s); "
                         "add system_*.mcap paths to populate power/host")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE,
                    help=f"download cache (default: {DEFAULT_CACHE})")
    ap.add_argument("--save", metavar="OUT.rrd",
                    help="write a self-contained .rrd (data + dashboard) "
                         "instead of spawning the viewer; needs exactly one "
                         "session")
    args = ap.parse_args(argv)

    if args.save and len(args.sessions) != 1:
        ap.error("--save takes exactly one session")
    paths = [_resolve(s, args.cache_dir) for s in args.sessions]
    if shutil.which("rerun") is None and not args.save:
        print("warning: `rerun` viewer not on PATH; spawn may fail",
              file=sys.stderr)

    blueprint = navd_blueprint.build()
    for path in paths:
        # One recording per session, all under the dashboard's app id.
        rec = rr.RecordingStream(navd_blueprint.NAME)
        if args.save:
            rec.save(args.save, default_blueprint=blueprint)
        else:
            rec.spawn(default_blueprint=blueprint)
            # The viewer persists a per-app-id blueprint and reuses it, so the
            # spawn-time default is ignored once one is cached — a blueprint
            # edit (e.g. new IMU panels) would not show. Send it explicitly and
            # make it the active/default so every new session picks it up.
            rec.send_blueprint(blueprint, make_active=True, make_default=True)
        print(f"loading {path}")
        rec.log_file_from_path(path)
        rec.flush()
        rec.disconnect()
    if args.save:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
