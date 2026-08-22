"""Compare the pre-registered structural arm with the frozen lambda=0 arm.

The comparison consumes per-row predictions, not result summaries. H2 reuses
the study's paired PDF-cluster bootstrap exactly: one PDF resample is shared by
both arms and all three seeds. H1 is descriptive by pre-registration and is
therefore reported without a p-value. Per-class F1 is retained for the safety
check, including the rare ``within_2_years`` and ``Not Clear`` classes.

    python -m analysis.structural_arm \
        --structural-root structural_arm \
        --probs-dir probs_structural \
        --out structural_arm/comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT, paired_delta
from analysis.load import METHODS, load_aligned, pdf_clusters, predictions_path
from analysis.metrics import weighted_macro_f1
from paper.data import REPO_ROOT, canonical_row_order, file_sha256, load_dev
from paper.labels import EVAL_FIELDS, FIELDS, INVALID_STATE_ID, tuple_to_state_id
from paper.train_config import PROTOCOLS, SEEDS

H1_METHOD = "M0"
H2_METHOD = "M1"
PRIMARY_PROTOCOL = "pdf_group"
NAMED_SAFETY_CLASSES = (
    ("verification_timeline", "within_2_years"),
    ("evidence_quality", "Not Clear"),
)


def _state_ids(codes: np.ndarray) -> np.ndarray:
    names = [[EVAL_FIELDS[field][code] for field, code in zip(FIELDS, row)]
             for row in codes]
    return np.array([tuple_to_state_id(*row) for row in names], dtype=np.int64)


def _invalid_rate(pair: tuple[np.ndarray, np.ndarray]) -> float:
    return float((_state_ids(pair[1]) == INVALID_STATE_ID).mean())


def _per_class_f1(gold: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for j, field in enumerate(FIELDS):
        labels = EVAL_FIELDS[field]
        gold_counts = np.bincount(gold[:, j], minlength=len(labels))
        pred_counts = np.bincount(pred[:, j], minlength=len(labels))
        tp = np.bincount(gold[gold[:, j] == pred[:, j], j], minlength=len(labels))
        denom = gold_counts + pred_counts
        scores = np.divide(
            2.0 * tp, denom, out=np.zeros(len(labels), dtype=float), where=denom > 0,
        )
        out[field] = {label: float(scores[i]) for i, label in enumerate(labels)}
    return out


def _mean_per_class(sets_by_seed) -> dict:
    per_seed = [_per_class_f1(gold, pred) for gold, pred in sets_by_seed]
    return {
        field: {
            label: float(np.mean([row[field][label] for row in per_seed]))
            for label in EVAL_FIELDS[field]
        }
        for field in FIELDS
    }


def _paired_sets(protocol, method, order, baseline_root, structural_root, seeds):
    baseline = [load_aligned(protocol, seed, method, order, baseline_root)
                for seed in seeds]
    structural = [load_aligned(protocol, seed, method, order, structural_root)
                  for seed in seeds]
    for (gold_b, _), (gold_s, _) in zip(baseline, structural):
        if not np.array_equal(gold_b, gold_s):
            raise ValueError(f"{protocol} {method}: arms disagree on aligned gold")
    return baseline, structural


def _protocol_comparison(protocol, order, baseline_root, structural_root, seeds):
    baseline_m0, structural_m0 = _paired_sets(
        protocol, H1_METHOD, order, baseline_root, structural_root, seeds,
    )
    baseline_m1, structural_m1 = _paired_sets(
        protocol, H2_METHOD, order, baseline_root, structural_root, seeds,
    )

    invalid_b = [_invalid_rate(pair) for pair in baseline_m0]
    invalid_s = [_invalid_rate(pair) for pair in structural_m0]
    score_b = [weighted_macro_f1(*pair) for pair in baseline_m1]
    score_s = [weighted_macro_f1(*pair) for pair in structural_m1]
    class_b = _mean_per_class(baseline_m1)
    class_s = _mean_per_class(structural_m1)

    all_methods = {}
    for method in METHODS:
        if method == H1_METHOD:
            baseline_sets, structural_sets = baseline_m0, structural_m0
        elif method == H2_METHOD:
            baseline_sets, structural_sets = baseline_m1, structural_m1
        else:
            baseline_sets, structural_sets = _paired_sets(
                protocol, method, order, baseline_root, structural_root, seeds,
            )
        baseline_scores = [weighted_macro_f1(*pair) for pair in baseline_sets]
        structural_scores = [weighted_macro_f1(*pair) for pair in structural_sets]
        all_methods[method] = {
            "baseline_per_seed": baseline_scores,
            "structural_per_seed": structural_scores,
            "baseline_mean": float(np.mean(baseline_scores)),
            "structural_mean": float(np.mean(structural_scores)),
            "delta": float(np.mean(structural_scores) - np.mean(baseline_scores)),
            "structural_sample_std": float(np.std(structural_scores, ddof=1)),
        }

    return {
        "all_method_weighted_macro_f1": all_methods,
        "h1_invalid_tuple_rate": {
            "method": H1_METHOD,
            "baseline_per_seed": invalid_b,
            "structural_per_seed": invalid_s,
            "baseline_mean": float(np.mean(invalid_b)),
            "structural_mean": float(np.mean(invalid_s)),
            "delta": float(np.mean(invalid_s) - np.mean(invalid_b)),
            "relative_reduction": float(1.0 - np.mean(invalid_s) / np.mean(invalid_b)),
            "lower_in_every_seed": all(s < b for b, s in zip(invalid_b, invalid_s)),
            "inference": "descriptive_only",
        },
        "h2_weighted_macro_f1": {
            "method": H2_METHOD,
            "baseline_per_seed": score_b,
            "structural_per_seed": score_s,
            "baseline_mean": float(np.mean(score_b)),
            "structural_mean": float(np.mean(score_s)),
            "delta": float(np.mean(score_s) - np.mean(score_b)),
        },
        "m1_per_class_f1": {
            field: {
                label: {
                    "baseline_mean": class_b[field][label],
                    "structural_mean": class_s[field][label],
                    "delta": class_s[field][label] - class_b[field][label],
                }
                for label in EVAL_FIELDS[field]
            }
            for field in FIELDS
        },
        "_m1_sets": (baseline_m1, structural_m1),
    }


def compare_arms(baseline_root, structural_root, *, n_boot=N_BOOT,
                 bootstrap_seed=BOOTSTRAP_SEED, seeds=SEEDS, dev=None) -> dict:
    """Return the pre-registered H1/H2 comparison and safety diagnostics."""
    dev = dev if dev is not None else load_dev()
    order = canonical_row_order(dev)
    clusters = pdf_clusters(order, dev)

    protocols = {}
    m1_sets = {}
    for protocol in PROTOCOLS:
        row = _protocol_comparison(
            protocol, order, baseline_root, structural_root, seeds,
        )
        m1_sets[protocol] = row.pop("_m1_sets")
        protocols[protocol] = row

    baseline_m1, structural_m1 = m1_sets[PRIMARY_PROTOCOL]
    h2 = paired_delta(
        structural_m1, baseline_m1, clusters,
        n_boot=n_boot, seed=bootstrap_seed,
    )
    h2.update({
        "protocol": PRIMARY_PROTOCOL,
        "contrast": "structural M1 - baseline M1",
        "alpha": 0.05,
        "multiplicity_adjustment": None,
        "supported": h2["delta"] > 0 and h2["p_value"] < 0.05,
    })

    named = {}
    classes = protocols[PRIMARY_PROTOCOL]["m1_per_class_f1"]
    for field, label in NAMED_SAFETY_CLASSES:
        named[f"{field}:{label}"] = classes[field][label]

    return {
        "design": {
            "baseline": "lambda=0",
            "structural": "selected lambda arm",
            "seeds": list(seeds),
            "h1_method": H1_METHOD,
            "h2_method": H2_METHOD,
            "bootstrap_unit": "pdf_url",
            "bootstrap_seed": bootstrap_seed,
        },
        "protocols": protocols,
        "h1": {
            "inference": "descriptive_only",
            "direction_observed_in_all_protocol_seed_pairs": all(
                protocols[p]["h1_invalid_tuple_rate"]["lower_in_every_seed"]
                for p in PROTOCOLS
            ),
        },
        "h2": h2,
        "safety_check": {
            "protocol": PRIMARY_PROTOCOL,
            "method": H2_METHOD,
            "named_rare_classes": named,
            "named_rare_class_decline": any(row["delta"] < 0 for row in named.values()),
        },
    }


def _selected_lambda(probs_dir: Path) -> float:
    values = set()
    metas = sorted(Path(probs_dir).glob("*/meta.json"))
    if not metas:
        raise ValueError(f"no structural bundles under {probs_dir}")
    for path in metas:
        with open(path, encoding="utf-8") as f:
            values.add(json.load(f).get("structure_lambda", 0.0))
    if len(values) != 1:
        raise ValueError(f"structural bundles mix lambdas: {sorted(values)}")
    return float(values.pop())


def _input_hashes(baseline_root, structural_root, seeds) -> dict:
    paths = {}
    for arm, root in (("baseline", baseline_root), ("structural", structural_root)):
        for protocol in PROTOCOLS:
            for seed in seeds:
                for method in METHODS:
                    path = predictions_path(protocol, seed, method, root)
                    paths[f"{arm}/{path.name}"] = file_sha256(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--structural-root", type=Path, default=REPO_ROOT / "structural_arm")
    ap.add_argument("--probs-dir", type=Path, default=REPO_ROOT / "probs_structural")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "structural_arm" / "comparison.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    report = compare_arms(
        args.baseline_root, args.structural_root,
        n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed,
    )
    report["design"]["structure_lambda"] = _selected_lambda(args.probs_dir)
    report["input_sha256"] = _input_hashes(
        args.baseline_root, args.structural_root, SEEDS,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"H1 lower in all pairs: "
          f"{report['h1']['direction_observed_in_all_protocol_seed_pairs']}")
    h2 = report["h2"]
    print(f"H2 delta={h2['delta']:.6f} "
          f"95% CI [{h2['ci_low']:.6f}, {h2['ci_high']:.6f}] "
          f"p={h2['p_value']:.4f} supported={h2['supported']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
