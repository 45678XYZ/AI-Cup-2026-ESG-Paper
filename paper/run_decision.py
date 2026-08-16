"""Run the real M0-M6 decision study from frozen probability bundles.

    python -m paper.run_decision
    python -m paper.run_decision --protocols pdf_group --seeds 42

Each (protocol, seed) consumes five rotations, fits decision parameters only on
each rotation's Calibration partition, concatenates the five Test partitions,
and computes one score over all 2,000 rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.artifacts import prediction_row, write_predictions
from paper.calibration import apply_biases, fit_biases, log_scores
from paper.data import REPO_ROOT, index_by_id, load_dev
from paper.decoder import DEFAULT_ALPHA, decode
from paper.evaluate import build_results, write_results
from paper.labels import FIELDS
from paper.projection import independent_argmax, project
from paper.train_config import PROTOCOLS, SEEDS
from paper.validate import validate_predictions, validate_probs_bundle, validate_probs_run

METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
BIAS_MODE = {"M0": None, "M1": None, "M2": "global", "M3": "conditional",
             "M4": None, "M5": "global", "M6": "conditional"}
OUTPUT_RULE = {"M0": "independent", "M1": "projection", "M2": "projection",
               "M3": "projection", "M4": "decoder", "M5": "decoder", "M6": "decoder"}


def _load_partition(bundle, partition):
    return {field: np.load(bundle / f"{partition}_{field}.npy") for field in FIELDS}


def _decide(method, probs, fitted):
    mode, rule = BIAS_MODE[method], OUTPUT_RULE[method]
    scores = apply_biases(probs, fitted[mode][0]) if mode else log_scores(probs)
    if rule == "independent":
        return independent_argmax(probs)
    if rule == "projection":
        return project(scores)
    return decode(scores, alpha=DEFAULT_ALPHA)


def run_one(protocol, seed, *, methods=METHODS, probs_dir=REPO_ROOT / "probs",
            predictions_dir=REPO_ROOT / "predictions", results_dir=REPO_ROOT / "results",
            overwrite=False):
    methods = tuple(methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    split_path = REPO_ROOT / "splits" / f"{protocol}_seed{seed}.json"
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    rows = load_dev()
    by_id = index_by_id(rows)
    bundles = [Path(probs_dir) / f"{protocol}_seed{seed}_r{k}" for k in range(5)]

    problems = []
    for bundle in bundles:
        problems += validate_probs_bundle(bundle, split=split)
    problems += validate_probs_run(bundles)
    if problems:
        raise ValueError("invalid input probability bundles:\n  " + "\n  ".join(problems))

    output_paths = [
        Path(predictions_dir) / f"{protocol}_seed{seed}_{method}.csv.gz" for method in methods
    ] + [Path(results_dir) / f"{protocol}_seed{seed}_{method}.json" for method in methods]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(existing))

    records = {method: [] for method in methods}
    decision_params = {method: {} for method in methods}

    for rot, bundle in zip(split["rotations"], bundles):
        calibration_probs = _load_partition(bundle, "calibration")
        test_probs = _load_partition(bundle, "test")
        calibration_rows = [by_id[row_id] for row_id in rot["calibration_ids"]]
        test_rows = [by_id[row_id] for row_id in rot["test_ids"]]

        needed_modes = {BIAS_MODE[m] for m in methods} - {None}
        fitted = {
            mode: fit_biases(
                calibration_probs, calibration_rows, partition="calibration", mode=mode,
            )
            for mode in needed_modes
        }

        for method in methods:
            predictions = _decide(method, test_probs, fitted)
            records[method] += [
                prediction_row(row, rot["k"], pred)
                for row, pred in zip(test_rows, predictions)
            ]
            mode = BIAS_MODE[method]
            fit_meta = fitted[mode][1] if mode else None
            decision_params[method][str(rot["k"])] = {
                "calibration_mode": mode,
                "calibration_biases": fitted[mode][0] if mode else None,
                "fallback_applied": fit_meta["fallback_applied"] if mode else {},
                "structural_pinned": fit_meta["structural_pinned"] if mode else {},
                "optimizer": ({k: fit_meta[k] for k in (
                    "objective", "grid", "max_passes", "calibration_objective"
                )} if mode else None),
                "decoder": ({"alpha": list(DEFAULT_ALPHA), "mode": "fixed"}
                            if OUTPUT_RULE[method] == "decoder" else None),
            }

    written = []
    for method in methods:
        pred_path = Path(predictions_dir) / f"{protocol}_seed{seed}_{method}.csv.gz"
        write_predictions(pred_path, records[method])
        pred_problems = validate_predictions(
            pred_path, rows=rows, split=split, method=method,
        )
        if pred_problems:
            raise ValueError("generated invalid predictions:\n  " + "\n  ".join(pred_problems))

        result = build_results(
            records[method], protocol=protocol, seed=seed, method=method,
            predictions_path=pred_path, data_checksum=split["data_checksum"],
            decision_params=decision_params[method],
        )
        result_path = write_results(
            Path(results_dir) / f"{protocol}_seed{seed}_{method}.json", result,
        )
        written.append((pred_path, result_path))
        print(f"{protocol} seed{seed} {method}: "
              f"weighted_macro_f1={result['weighted_macro_f1']:.4f} "
              f"invalid={result['invalid_tuple_rate']:.3%}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=list(PROTOCOLS))
    ap.add_argument("--seeds", nargs="+", type=int, choices=list(SEEDS), default=list(SEEDS))
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    ap.add_argument("--probs-dir", type=Path, default=REPO_ROOT / "probs")
    ap.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "predictions")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for protocol in args.protocols:
        for seed in args.seeds:
            run_one(
                protocol, seed, methods=args.methods, probs_dir=args.probs_dir,
                predictions_dir=args.predictions_dir, results_dir=args.results_dir,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
