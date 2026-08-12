"""Synthetic probability fixtures (interface contract §3, "A's alternative input").

Lets the decision pipeline be built and smoke-tested before any GPU output
exists: the fixtures are contract-shaped, so switching to B's real
probabilities is a path change and nothing else.

``concentration`` controls how much mass sits on the gold label, which makes
the expected behaviour of M0-M6 analytically predictable:

    concentration=1.0   near one-hot on gold -> every method should score ~1.0
                        and produce no invalid tuples
    concentration=0.0   uniform noise -> methods are separated only by their
                        decision rule, so hierarchy violations in M0 become
                        visible while M1-M6 must stay at 0
    in between          gold-favouring but confusable, the realistic case

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


def _probs_for(rows, field, rng, concentration):
    labels = EVAL_FIELDS[field]
    n, c = len(rows), len(labels)
    probs = rng.random((n, c)).astype(np.float64) + 1e-3
    gold = np.array([LABEL2ID[field][r[field]] for r in rows])
    # Push mass onto the gold column; concentration=1 is effectively one-hot.
    probs[np.arange(n), gold] += concentration * c * 10.0
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def make_rotation_fixture(split, k, out_dir, concentration=0.6, rows=None):
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
    ap.add_argument("--rotations", nargs="+", type=int, default=[0])
    ap.add_argument("--concentration", type=float, default=0.6)
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
