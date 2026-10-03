"""Deprecated compatibility wrapper around the catalog downloader.

Kept so existing invocations keep working:

    python -m bebop_vision.download_sam3            # sam3 + sam3.1
    python -m bebop_vision.download_sam3 sam3.1     # one model

New code should use the catalog-driven downloader:

    python -m bebop_vision.download_model <id>
"""

import sys

from .download_model import main as _download


def main() -> int:
    ids = sys.argv[1:] or ["sam3", "sam3.1"]
    return _download(ids)


if __name__ == "__main__":
    raise SystemExit(main())
