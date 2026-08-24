"""The pre-registered lambda-selection criterion, as code rather than prose.

``docs/preregistration/pre_registration_structural_training.md`` §4 fixes how the structural
arm's lambda is chosen. A criterion that has to be hand-rolled at selection
time is not pre-registered in any useful sense -- whoever runs it decides what
it meant. This module is that criterion, so the choice is reproducible and the
sweep can be audited afterwards.

**It cannot read the Test partition.** Only ``calibration_*.npy`` and
``meta.json:calibration_ids`` are opened; the test arrays sitting in the same
bundle directory are never touched. That is the property the whole selection
rests on, and ``tests/test_select_lambda.py`` asserts it by patching np.load.

The score is the official weighted macro-F1 of **independent per-field argmax**
-- no projection, no decoding, no bias. The point of the sweep is to compare
the probabilities each lambda produces, and interposing a decision rule would
confound that with how well the rule repairs them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper.data import REPO_ROOT, index_by_id, load_dev
from paper.labels import EVAL_FIELDS, FIELDS
from paper.score import compute_weighted_macro_f1
from paper.validate import recipe_value


def calibration_score(bundle_dir, by_id) -> float:
    """Weighted macro-F1 of independent argmax on this rotation's Calibration rows."""
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["calibration_ids"]

    probs = {f: np.load(bundle_dir / f"calibration_{f}.npy") for f in FIELDS}
    for f in FIELDS:
        if len(probs[f]) != len(ids):
            raise ValueError(
                f"{bundle_dir.name}: {len(probs[f])} rows of {f} for {len(ids)} "
                "calibration ids"
            )
    pred = [{f: EVAL_FIELDS[f][int(np.argmax(probs[f][n]))] for f in FIELDS}
            for n in range(len(ids))]
    gold = [{f: by_id[i][f] for f in FIELDS} for i in ids]
    return compute_weighted_macro_f1(gold, pred)


def sweep_scores(probs_dir, by_id=None) -> dict:
    """``{structure_lambda: {"rotations": n, "mean": score, "scores": [...]}}``.

    Groups whatever bundles are present by the lambda their meta records, so a
    sweep directory and the frozen study's directory are read the same way.
    """
    by_id = by_id if by_id is not None else index_by_id(load_dev())
    grouped = defaultdict(list)
    # Recursive: the sweep writes each lambda to its own subdirectory, because
    # all three produce the same five bundle names and a flat directory would
    # have them overwrite each other.
    for d in sorted(p.parent for p in Path(probs_dir).rglob("meta.json")):
        with open(d / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        label = str(d.relative_to(Path(probs_dir)))
        grouped[recipe_value(meta, "structure_lambda")].append(
            (label, calibration_score(d, by_id))
        )
    return {
        lam: {
            "rotations": len(pairs),
            "mean": float(np.mean([s for _, s in pairs])),
            "scores": dict(pairs),
        }
        for lam, pairs in sorted(grouped.items(), key=lambda kv: (kv[0] is None, kv[0]))
    }


# Below this, the sweep has not resolved lambda and the pre-registration's
# fallback applies (§4 step 4): take the grid's median rather than the winner.
INDISTINGUISHABLE = 0.002


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probs-dir", type=Path, default=REPO_ROOT / "runs/lambda_sweep")
    args = ap.parse_args()

    table = sweep_scores(args.probs_dir)
    if not table:
        raise SystemExit(f"no bundles with a meta.json under {args.probs_dir}")

    print(f"{'structure_lambda':>18s} {'rotations':>10s} {'calibration wF1':>17s}")
    for lam, row in table.items():
        print(f"{str(lam):>18s} {row['rotations']:>10d} {row['mean']:>17.4f}")

    means = {lam: row["mean"] for lam, row in table.items()}
    spread = max(means.values()) - min(means.values())
    best = max(means, key=means.get)
    print(f"\nspread {spread:.4f}")
    if len(means) > 1 and spread < INDISTINGUISHABLE:
        print(f"below {INDISTINGUISHABLE}: the sweep did not resolve lambda. "
              "Pre-registration §4 step 4 applies -- take the grid's median and "
              "record in the paper that lambda was not resolved.")
    else:
        print(f"selected lambda = {best}")
    print("\nRecord the outcome in docs/preregistration/pre_registration_structural_training.md §8.")


if __name__ == "__main__":
    main()
