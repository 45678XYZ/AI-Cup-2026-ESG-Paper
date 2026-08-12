"""Synthetic probability fixtures (interface contract §3, "A's alternative input").

Lets the decision pipeline be built and smoke-tested before any GPU output
exists: the fixtures are contract-shaped, so switching to B's real
probabilities is a path change and nothing else.

``concentration`` is the weight placed on a one-hot at the gold label, the rest
going to uniform noise:

    0.0          pure noise -> the decision rule is the only thing separating
                 the methods, so M0's hierarchy violations become visible while
                 M1-M6 must stay at zero
    0 < c < 0.5  gold-favouring but confusable: the band that actually exercises
                 a decision rule, and the only band worth debugging in
    c >= 0.5     gold wins every row *by construction* -- the noise component
                 can contribute at most (1 - c), so no other class can overtake
                 it. A perfect oracle: every method scores 1.0 and no method can
                 be told apart from any other.

Measured on rotation 0 of pdf_group_seed42, as a guide when picking a value:

    c     rows where M0 breaks the hierarchy
    0.0   86%
    0.2   63%
    0.4   13%
    0.5+   0%

    python -m contracts.make_fixtures --out contracts/examples/probs

Fixtures are synthetic by construction and must never reach a results table:
main-table numbers come from cross-fitted probabilities only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from paper.artifacts import write_probs_bundle
from paper.data import REPO_ROOT, load_dev
from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID

# Inside the useful band and near its confident end: the gold label usually
# wins, but often enough loses that M0 breaks the hierarchy on ~13% of rows,
# which is what gives the decision rules something to differ about.
DEFAULT_CONCENTRATION = 0.4


def _probs_for(rows, field, rng, concentration):
    """Linear mix of uniform noise and a one-hot on gold.

    ``(1 - c) * noise + c * onehot``. Both components are already distributions,
    so every row sums to 1 without a renormalisation step, and ``concentration``
    behaves as an actual dial: the probability of the gold label winning rises
    smoothly from chance at 0 to certainty at 1.

    An earlier version added ``concentration * n_classes * 10`` to the gold
    column of unnormalised noise. That swamped the noise (which peaks near 1.0)
    at any setting above ~0.05, so the knob had two positions -- pure noise, or
    a perfect oracle -- and nothing in between could exercise a decision rule.
    """
    labels = EVAL_FIELDS[field]
    n, c = len(rows), len(labels)

    noise = rng.random((n, c))
    noise /= noise.sum(axis=1, keepdims=True)

    onehot = np.zeros((n, c))
    onehot[np.arange(n), [LABEL2ID[field][r[field]] for r in rows]] = 1.0

    return ((1.0 - concentration) * noise + concentration * onehot).astype(np.float32)


def make_rotation_fixture(split, k, out_dir, concentration=DEFAULT_CONCENTRATION, rows=None):
    """Write one probs/{protocol}_seed{seed}_r{k}/ bundle."""
    rows = rows if rows is not None else load_dev()
    by_id = {r["id"]: r for r in rows}
    rot = next(r for r in split["rotations"] if r["k"] == k)
    # Seed from a digest, not from hash(): hash() of a tuple containing a str
    # is salted per process, so the "same" fixture would differ on every run
    # and the checksums recorded in meta.json could never be reproduced.
    material = f"{split['protocol']}:{split['seed']}:{k}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))

    # Row order follows the split manifest, exactly as the real driver does.
    probs = {
        partition: {
            field: _probs_for([by_id[i] for i in rot[f"{partition}_ids"]], field, rng, concentration)
            for field in FIELDS
        }
        for partition in ("calibration", "test")
    }

    # Same writer as paper/run_training.py, so a fixture and a GPU run cannot
    # differ in structure -- only in where the numbers came from.
    return write_probs_bundle(
        out_dir, split, k, probs,
        extra_meta={
            "model_name": "SYNTHETIC-FIXTURE",
            "model_revision": "synthetic",
            "checkpoint_rule": "n/a",
            "synthetic": True,
            "concentration": concentration,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic probability fixtures.")
    ap.add_argument("--protocol", default="pdf_group")
    ap.add_argument("--seed", type=int, default=42)
    # All five by default: the concatenation path (five test partitions -> one
    # 2,000-row score) is where duplicate-id and row-count bugs live, and a
    # single rotation cannot exercise it.
    ap.add_argument("--rotations", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--concentration", type=float, default=DEFAULT_CONCENTRATION)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "contracts" / "examples" / "probs")
    args = ap.parse_args()

    split_path = REPO_ROOT / "splits" / f"{args.protocol}_seed{args.seed}.json"
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)

    rows = load_dev()
    for k in args.rotations:
        out = args.out / f"{args.protocol}_seed{args.seed}_r{k}"
        make_rotation_fixture(split, k, out, args.concentration, rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
