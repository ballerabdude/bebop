"""Placeholder Bebop robot application entrypoint.

Replace this with the real robot bring-up logic. `bebop-agent` runs this
container under the NVIDIA runtime so CUDA / NVENC / etc. are available.
"""

from __future__ import annotations

import signal
import sys
import time


def _install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)


def _shutdown(signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    print(f"[bebop-app] received signal {signum}, exiting", flush=True)
    sys.exit(0)


def main() -> None:
    _install_signal_handlers()
    print("[bebop-app] starting", flush=True)
    # Replace with your robot's actual main loop.
    while True:
        print("[bebop-app] heartbeat", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
