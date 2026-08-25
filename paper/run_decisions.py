"""Driver for the decision stage: probability bundles in, contract-3 files out.

    python -m paper.run_decisions --protocol pdf_group --seed 42
    python -m paper.run_decisions --protocol pdf_group --seed 42 --methods M0 M1
    python -m paper.run_decisions --protocol pdf_group --seed 42 \
        --probs-dir contracts/examples/probs --out-dir /tmp/smoke

This is the counterpart of ``paper/run_training.py`` on A's side of the
boundary, and it is where the study's central claim is enforced mechanically:
every method reads the *same* bundles and is scored on the *same* rows, because
one invocation runs all of them over one loaded set of probabilities.

The five rotations of a (protocol, seed) are concatenated into a single
corpus-wide file before anything is scored — never averaged (plan §4.2 rule 7),
because folds differ in which rare classes they contain and their F1 values are
not on a common scale.
"""

import argparse
from pathlib import Path

import numpy as np

from paper.artifacts import prediction_row, write_predictions
from paper.calibration import as_json, fit_biases
from paper.corpus import DEFAULT as DEFAULT_CORPUS
from paper.corpus import CORPORA, load_rows
from paper.data import REPO_ROOT, data_checksum, index_by_id, load_dev
from paper.evaluate import build_results, write_results
from paper.labels import FIELDS
from paper.methods import METHOD_IDS, METHODS, decide
from paper.train_config import N_FOLDS, SEEDS
from paper.validate import SPLITS_DIR, load_split, validate_probs_bundle, validate_probs_run


def load_bundle(bundle_dir) -> tuple[dict, dict]:
    """Return ``(meta, {"calibration": {...}, "test": {...}})`` for one rotation."""
    import json

    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    probs = {
        partition: {
            field: np.load(bundle_dir / f"{partition}_{field}.npy")
            for field in FIELDS
        }
        for partition in ("calibration", "test")
    }
    return meta, probs


def load_run(probs_dir, protocol, seed, split,
             splits_dir=SPLITS_DIR) -> list[tuple[dict, dict]]:
    """Load and check all five rotations of one (protocol, seed).

    Validation happens here rather than being left to the operator: these
    bundles are the only input the results depend on, and every failure the
    validator catches is one that would otherwise produce a plausible number.
    """
    dirs = [Path(probs_dir) / f"{protocol}_seed{seed}_r{k}" for k in range(N_FOLDS)]
    missing = [d.name for d in dirs if not (d / "meta.json").exists()]
    if missing:
        raise SystemExit(f"missing probability bundles: {missing}")

    # Keep the run-level coverage check on the exact split main loaded. This
    # may describe a 400-row English/French/Japanese corpus, a 500-row Korean
    # corpus, or the 2,000-row Chinese corpus; falling back to the validator's
    # default would silently check the wrong geometry.
    problems = validate_probs_run(dirs, splits_dir=splits_dir, split=split)
    for d in dirs:
        problems += validate_probs_bundle(d, split)
    if problems:
        raise SystemExit(
            "refusing to run on non-conforming bundles:\n  "
            + "\n  ".join(problems)
        )
    return [load_bundle(d) for d in dirs]


def run_method(method_id, run, rows, split, bias_cache=None) -> tuple[list[dict], dict]:
    """One method over all five rotations, concatenated in rotation order.

    ``bias_cache`` is keyed by ``(mode, rotation)`` and shared across methods by
    ``main``, so M2 and M5 are handed the *same* global bias object and M3 and
    M6 the same conditional one. The factorial reading of the results table
    rests on that identity, and this makes it structural rather than a
    consequence of the estimator happening to be deterministic.
    """
    method = METHODS[method_id]
    by_id = index_by_id(rows)
    records, decision_params = [], {}

    for meta, probs in run:
        rotation = meta["rotation"]
        # Biases are estimated on the Calibration partition only; the
        # uncalibrated methods get None.
        biases, fallback = None, None
        if method.calibration is not None:
            key = (method.calibration, rotation)
            if bias_cache is not None and key in bias_cache:
                biases, fallback = bias_cache[key]
            else:
                biases, fallback = fit_biases(
                    method.calibration, probs["calibration"],
                    meta["calibration_ids"],
                    next(r for r in split["rotations"] if r["k"] == rotation),
                    rows,
                )
                if bias_cache is not None:
                    bias_cache[key] = (biases, fallback)

        preds = decide(method, probs["test"], biases)
        test_ids = meta["test_ids"]
        if len(preds) != len(test_ids):
            raise ValueError(
                f"r{rotation}: {len(preds)} predictions for {len(test_ids)} rows"
            )
        records += [
            prediction_row(by_id[row_id], rotation, pred)
            for row_id, pred in zip(test_ids, preds)
        ]
        decision_params[str(rotation)] = {
            "calibration_biases": as_json(biases) if biases is not None else None,
            "fallback_applied": fallback,
            "decoder": ({"alpha": [1.0] * len(FIELDS), "mode": "fixed"}
                        if method.output_rule == "decoder" else None),
        }

    return records, decision_params


def _check_complete(records, rows, method_id) -> None:
    """Every row exactly once: the property that makes one score, not five."""
    ids = [r["id"] for r in records]
    expected = {r["id"] for r in rows}
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError(
            f"{method_id}: concatenated {len(ids)} rows covering "
            f"{len(set(ids))} distinct ids, expected {len(expected)} each once"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", required=True, choices=["pdf_group", "row_strat"])
    ap.add_argument("--seed", required=True, type=int, choices=list(SEEDS))
    ap.add_argument("--methods", nargs="+", default=METHOD_IDS, choices=METHOD_IDS)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
                    help="which labelled rows the bundles were fit on; supplies "
                         "the defaults for --probs-dir and --splits-dir")
    ap.add_argument("--probs-dir", type=Path, default=None)
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.probs_dir is None:
        args.probs_dir = REPO_ROOT / CORPORA[args.corpus]["probs_dir"]
    if args.splits_dir is None:
        args.splits_dir = REPO_ROOT / CORPORA[args.corpus]["splits_dir"]
    if args.out_dir is None:
        # Beside the bundles it scored, not at a fixed root. A decision run
        # names its output {protocol}_seed{seed}_{method}.csv.gz whatever it
        # was fit on, so seven arms sharing one output directory would each
        # overwrite the last, and an English run defaulted to the repository
        # root would overwrite the frozen study's predictions file by file.
        args.out_dir = (REPO_ROOT / CORPORA[args.corpus]["decisions_root"]
                        if args.corpus == DEFAULT_CORPUS
                        else args.probs_dir.parent)
        print(f"writing decisions to {args.out_dir}")

    rows = load_rows(args.corpus)
    checksum = data_checksum(rows)
    split = load_split(args.protocol, args.seed, args.splits_dir)
    if split["data_checksum"] != checksum:
        raise SystemExit(
            f"{args.splits_dir} was built against different data; refusing to "
            f"run.\n--corpus is {args.corpus!r}."
        )

    # Loaded once and shared by every method: this is what "identical
    # probabilities on identical rows" means operationally.
    run = load_run(args.probs_dir, args.protocol, args.seed, split,
                   splits_dir=args.splits_dir)
    print(f"{args.protocol} seed{args.seed}: {len(run)} rotations validated", flush=True)

    bias_cache: dict = {}
    for method_id in args.methods:
        records, decision_params = run_method(method_id, run, rows, split, bias_cache)
        _check_complete(records, rows, method_id)

        stem = f"{args.protocol}_seed{args.seed}_{method_id}"
        pred_path = write_predictions(
            Path(args.out_dir) / "predictions" / f"{stem}.csv.gz", records,
        )
        results = build_results(
            records,
            protocol=args.protocol, seed=args.seed, method=method_id,
            predictions_path=pred_path, data_checksum=checksum,
            decision_params=decision_params,
        )
        write_results(Path(args.out_dir) / "results" / f"{stem}.json", results)
        print(f"  {method_id}: weighted_macro_f1={results['weighted_macro_f1']:.4f} "
              f"tuple_acc={results['tuple_exact_match']:.4f} "
              f"invalid={results['invalid_tuple_rate']:.3%}", flush=True)

    print(f"\nwritten to {args.out_dir}/predictions/ and {args.out_dir}/results/")


if __name__ == "__main__":
    main()
