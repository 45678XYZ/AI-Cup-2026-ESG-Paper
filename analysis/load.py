"""Load prediction sets onto one shared row order.

Every one of the 42 prediction files covers the same 2,000 rows in its own
rotation order. Reindexing all of them to ``canonical_row_order`` is what makes
the bootstrap paired: a single PDF resample becomes an index array valid for
every method, every seed and both protocols at once, so two methods are always
compared on identical rows and Table 3 can compare protocols on a shared
resample.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.metrics import encode
from paper.artifacts import read_predictions
from paper.data import REPO_ROOT, load_dev

METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")

# W2 works against the synthetic examples; W3 switches to the real thing by
# changing this one argument, which is the whole point of contract section 4.
EXAMPLES_ROOT = REPO_ROOT / "contracts" / "examples"
REAL_ROOT = REPO_ROOT


def predictions_path(protocol, seed, method, root) -> Path:
    return Path(root) / "predictions" / f"{protocol}_seed{seed}_{method}.csv.gz"


def load_aligned(protocol, seed, method, order, root):
    """``(gold, pred)`` class-id arrays for one method, in ``order``."""
    path = predictions_path(protocol, seed, method, root)
    records = read_predictions(path)
    if len(records) != len(order):
        raise ValueError(f"{path.name}: {len(records)} rows, expected {len(order)}")
    return encode(records, order=order)


def pdf_clusters(order, dev=None):
    """Row positions grouped by source PDF -- the bootstrap's resampling unit.

    Positions refer to ``order``, so the returned arrays index straight into
    the ``gold``/``pred`` arrays of every loaded method.
    """
    dev = dev if dev is not None else load_dev()
    by_id = {r["id"]: r for r in dev}
    groups = defaultdict(list)
    for position, row_id in enumerate(order):
        groups[by_id[row_id]["pdf_url"]].append(position)
    return [np.array(groups[pdf], dtype=np.int64) for pdf in sorted(groups)]


def load_all(protocol, seed, order, root, methods=METHODS):
    """``{method: (gold, pred)}`` for one (protocol, seed)."""
    return {m: load_aligned(protocol, seed, m, order, root) for m in methods}
