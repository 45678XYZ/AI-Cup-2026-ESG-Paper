"""Summarize the pre-registered ML-Promise English replication.

The report is derived from per-row predictions, not copied from result JSON.
It includes both the AI CUP-weighted aggregate used for cross-arm comparability
and the equal-weight companion metric, plus the replication's only inferential
test: report-cluster bootstrap of tuple accuracy M1-M0 on the RoBERTa-large
lambda=0 document-disjoint arm.

    python -m analysis.english_replication
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT, paired_delta
from analysis.load import METHODS, load_aligned, pdf_clusters, predictions_path
from analysis.metrics import (
    field_macro_f1,
    tuple_accuracy,
    unweighted_macro_f1,
    weighted_macro_f1,
)
from paper.data import REPO_ROOT, data_checksum, file_sha256
from paper.data_en import canonical_row_order, load_english
from paper.labels import EVAL_FIELDS, FIELD_ALIAS, FIELDS, LABEL2ID, STATES
from paper.train_config import SEEDS

BACKBONES = {
    "roberta_large": ("roberta-large", ("pdf_group", "row_strat")),
    "deberta_v3_large": ("microsoft/deberta-v3-large", ("pdf_group",)),
    "electra_large_discriminator": (
        "google/electra-large-discriminator", ("pdf_group",),
    ),
    "roberta_base": ("roberta-base", ("pdf_group",)),
}
LAMBDAS = (0.0, 0.3)
PRIMARY_BACKBONE = "roberta_large"
PRIMARY_PROTOCOL = "pdf_group"

LEGAL_CODES = {
    tuple(
        LABEL2ID[field][getattr(state, FIELD_ALIAS[field])]
        for field in FIELDS
    )
    for state in STATES
}


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _invalid_rate(pred: np.ndarray) -> float:
    return float(np.mean([tuple(map(int, row)) not in LEGAL_CODES for row in pred]))


def _sets(root, protocol, method, order, seeds):
    return [load_aligned(protocol, seed, method, order, root) for seed in seeds]


def summarize_method(root, protocol, method, order, seeds=SEEDS) -> dict:
    """Descriptive metrics for one method, each scored once per full seed."""
    sets = _sets(root, protocol, method, order, seeds)
    per_seed = []
    for seed, (gold, pred) in zip(seeds, sets):
        per_seed.append({
            "seed": int(seed),
            "weighted_macro_f1": weighted_macro_f1(gold, pred),
            "unweighted_macro_f1": unweighted_macro_f1(gold, pred),
            "per_field_macro_f1": field_macro_f1(gold, pred),
            "tuple_accuracy": tuple_accuracy(gold, pred),
            "invalid_tuple_rate": _invalid_rate(pred),
        })

    out = {"per_seed": per_seed}
    for metric in (
        "weighted_macro_f1", "unweighted_macro_f1",
        "tuple_accuracy", "invalid_tuple_rate",
    ):
        mean, std = _mean_std([row[metric] for row in per_seed])
        out[f"{metric}_mean"] = mean
        out[f"{metric}_sample_std"] = std
    out["per_field_macro_f1_mean"] = {
        field: float(np.mean([
            row["per_field_macro_f1"][field] for row in per_seed
        ]))
        for field in FIELDS
    }
    return out


def _runtime(probs_dir: Path, expected_count: int, expected_lambda: float) -> dict:
    paths = sorted(Path(probs_dir).glob("*/meta.json"))
    if len(paths) != expected_count:
        raise ValueError(f"{probs_dir}: {len(paths)} bundles, expected {expected_count}")
    metas = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    lambdas = {float(meta["structure_lambda"]) for meta in metas}
    if lambdas != {expected_lambda}:
        raise ValueError(f"{probs_dir}: structure lambdas are {sorted(lambdas)}")
    return {
        "bundle_count": len(metas),
        "model_names": sorted({meta["model_name"] for meta in metas}),
        "model_revisions": sorted({meta["model_revision"] for meta in metas}),
        "git_shas": sorted({meta["git_sha"] for meta in metas}),
        "hardware": sorted({meta["hardware"] for meta in metas}),
        "structure_lambda": expected_lambda,
        "total_train_seconds": float(sum(meta["train_seconds"] for meta in metas)),
    }


def _input_hashes(runs_root, seeds=SEEDS) -> dict:
    out = {}
    for slug, (_, protocols) in BACKBONES.items():
        for structure_lambda in LAMBDAS:
            root = Path(runs_root) / slug / f"lambda_{structure_lambda:.1f}"
            for protocol in protocols:
                for seed in seeds:
                    for method in METHODS:
                        path = predictions_path(protocol, seed, method, root)
                        out[str(path.relative_to(runs_root))] = file_sha256(path)
    return out


def _sign_pattern(summaries) -> dict:
    out = {}
    for slug in BACKBONES:
        by_lambda = {}
        for structure_lambda in LAMBDAS:
            methods = summaries[slug][f"lambda_{structure_lambda:.1f}"][PRIMARY_PROTOCOL]
            m0, m1 = methods["M0"], methods["M1"]
            by_lambda[f"lambda_{structure_lambda:.1f}"] = {
                "m0_invalid_tuple_rate": m0["invalid_tuple_rate_mean"],
                "m1_minus_m0_weighted_macro_f1":
                    m1["weighted_macro_f1_mean"] - m0["weighted_macro_f1_mean"],
                "m1_minus_m0_unweighted_macro_f1":
                    m1["unweighted_macro_f1_mean"] - m0["unweighted_macro_f1_mean"],
                "m1_minus_m0_tuple_accuracy":
                    m1["tuple_accuracy_mean"] - m0["tuple_accuracy_mean"],
                "weighted_sign": "positive" if m1["weighted_macro_f1_mean"]
                    > m0["weighted_macro_f1_mean"] else "non_positive",
                "tuple_sign": "positive" if m1["tuple_accuracy_mean"]
                    > m0["tuple_accuracy_mean"] else "non_positive",
            }
        out[slug] = by_lambda
    return out


def build_report(runs_root=REPO_ROOT / "runs_en", *, n_boot=N_BOOT,
                 bootstrap_seed=BOOTSTRAP_SEED, seeds=SEEDS, rows=None) -> dict:
    rows = rows if rows is not None else load_english()
    order = canonical_row_order(rows)
    clusters = pdf_clusters(order, rows)

    summaries, runtime = {}, {}
    for slug, (model_name, protocols) in BACKBONES.items():
        summaries[slug], runtime[slug] = {}, {}
        for structure_lambda in LAMBDAS:
            label = f"lambda_{structure_lambda:.1f}"
            root = Path(runs_root) / slug / label
            summaries[slug][label] = {
                protocol: {
                    method: summarize_method(root, protocol, method, order, seeds)
                    for method in METHODS
                }
                for protocol in protocols
            }
            runtime[slug][label] = _runtime(
                root / "probs", len(protocols) * len(seeds) * 5, structure_lambda,
            )
            if runtime[slug][label]["model_names"] != [model_name]:
                raise ValueError(
                    f"{slug}/{label}: model is {runtime[slug][label]['model_names']}, "
                    f"expected {model_name}"
                )

    primary = Path(runs_root) / PRIMARY_BACKBONE / "lambda_0.0"
    m1 = _sets(primary, PRIMARY_PROTOCOL, "M1", order, seeds)
    m0 = _sets(primary, PRIMARY_PROTOCOL, "M0", order, seeds)
    h_en3 = paired_delta(
        m1, m0, clusters, n_boot=n_boot,
        seed=bootstrap_seed, score=tuple_accuracy,
    )
    h_en3.update({
        "contrast": "M1-M0 tuple accuracy",
        "alternative": "greater_than_zero",
        "bootstrap_unit": "pdf_url",
        "n_clusters": len(clusters),
        "supported": h_en3["ci_low"] > 0,
        "multiplicity_adjustment": None,
    })

    signs = _sign_pattern(summaries)
    lambda0 = [row["lambda_0.0"] for row in signs.values()]
    all_method_rows = [
        summaries[slug][lam][protocol][method]
        for slug, (_, protocols) in BACKBONES.items()
        for lam in ("lambda_0.0", "lambda_0.3")
        for protocol in protocols
        for method in ("M1", "M4")
    ]

    return {
        "design": {
            "corpus": "ML-Promise English",
            "data_checksum": data_checksum(rows),
            "n_rows": len(rows),
            "n_report_clusters": len(clusters),
            "seeds": list(seeds),
            "methods": list(METHODS),
            "lambdas": list(LAMBDAS),
            "weighted_metric_label": "AI CUP weights applied to ML-Promise",
            "companion_metric": "unweighted macro-F1 (four fields at 0.25)",
            "bootstrap_seed": bootstrap_seed,
        },
        "hypotheses": {
            "H-EN1": {
                "inference": "descriptive_only",
                "m0_invalid_positive_for_every_lambda0_backbone":
                    all(row["m0_invalid_tuple_rate"] > 0 for row in lambda0),
                "per_backbone": {
                    slug: row["lambda_0.0"]["m0_invalid_tuple_rate"]
                    for slug, row in signs.items()
                },
            },
            "H-EN2": {
                "inference": "implementation_check",
                "methods": ["M1", "M4"],
                "max_invalid_tuple_rate": max(
                    row["invalid_tuple_rate_mean"] for row in all_method_rows
                ),
                "passed": all(
                    row["invalid_tuple_rate_mean"] == 0 for row in all_method_rows
                ),
            },
            "H-EN3": h_en3,
            "H-EN4": {
                "inference": "descriptive_sign_pattern",
                "lambda0_all_weighted_positive": all(
                    row["m1_minus_m0_weighted_macro_f1"] > 0 for row in lambda0
                ),
                "lambda0_all_tuple_positive": all(
                    row["m1_minus_m0_tuple_accuracy"] > 0 for row in lambda0
                ),
                "per_backbone": signs,
            },
        },
        "summaries": summaries,
        "runtime": runtime,
        "input_sha256": _input_hashes(Path(runs_root), seeds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs_en")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs_en" / "summary.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    report = build_report(
        args.runs_root, n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")

    h = report["hypotheses"]
    print(f"H-EN1 all lambda=0 M0 invalid positive: "
          f"{h['H-EN1']['m0_invalid_positive_for_every_lambda0_backbone']}")
    print(f"H-EN2 valid-by-construction check: {h['H-EN2']['passed']}")
    row = h["H-EN3"]
    print(f"H-EN3 tuple M1-M0={row['delta']:+.6f} "
          f"95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] "
          f"p(two-sided)={row['p_value']:.4f} supported={row['supported']}")
    print(f"H-EN4 lambda=0 weighted all positive: "
          f"{h['H-EN4']['lambda0_all_weighted_positive']}; "
          f"tuple all positive: {h['H-EN4']['lambda0_all_tuple_positive']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
