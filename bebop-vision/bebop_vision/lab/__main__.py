"""``python -m bebop_vision.lab`` entry point."""

import os
import sys

# Same rule as main.py: the Thor's OpenBLAS worker pool spin-waits after tiny
# matmuls; one thread each, set before numpy is imported anywhere.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from bebop_vision.lab.cli import main  # noqa: E402

sys.exit(main())
