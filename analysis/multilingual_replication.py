"""Summarize the French, Japanese and Korean ML-Promise replications.

The summaries are rebuilt from the per-row prediction files, not copied from
the result JSON.  That keeps the prediction rows as the metric source of truth
and makes the report reproducible from a clone, including for Korean: its
locally extracted report text is required for fitting, but not for scoring the
committed predictions against their committed split order.

    python -m analysis.multilingual_replication
"""

import argparse
import json
from pathlib import Path

from analysis.english_replication import _mean_std, summarize_method
from analysis.load import METHODS, predictions_path
from paper.corpus import arm_dir, splits_dir
from paper.data import REPO_ROOT, file_sha256
from paper.data_ml import provenance, release_rows
from paper.multilingual_config import LAMBDAS, LANGUAGES, MODELS, SEEDS


REPORT_VERSION = "1.0"
LANGUAGE_CODES = {"mlpromise_fr": "fr", "mlpromise_ja": "ja", "mlpromise_ko": "ko"}
LANGUAGE_NAMES = {"fr": "French", "ja": "Japanese", "ko": "Korean"}
CONTRASTS = (
    ("M1", "M0", "hierarchy legalisation"),
    ("M4", "M1", "17-state decoding versus projection"),
    ("M6", "M5", "conditional versus global decoder calibration"),
)


def _split_identity(corpus: str) -> tuple[list[str], str]:
    path = REPO_ROOT / splits_dir(corpus) / "pdf_group_seed42.json"
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    return split["canonical_row_order"], split["data_checksum"]


def _runtime(probs_dir: Path, expected_count: int, expected_lambda: float) -> dict:
    paths = sorted(probs_dir.glob("*/meta.json"))
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
        "amp_dtypes": sorted({meta.get("amp_dtype", "float16") for meta in metas}),
        "structure_lambda": expected_lambda,
        "total_train_seconds": float(sum(meta["train_seconds"] for meta in metas)),
    }


def _contrast(methods: dict, first: str, second: str) -> dict:
    """Paired seed deltas for one pre-registered within-arm contrast."""
    out = {}
    for metric in ("weighted_macro_f1", "tuple_accuracy", "invalid_tuple_rate"):
        left = {row["seed"]: row[metric] for row in methods[first]["per_seed"]}
        right = {row["seed"]: row[metric] for row in methods[second]["per_seed"]}
        if left.keys() != right.keys():
            raise ValueError(f"{first} and {second} do not contain the same seeds")
        per_seed = [left[seed] - right[seed] for seed in sorted(left)]
        mean, std = _mean_std(per_seed)
        out[metric] = {
            "per_seed_delta": per_seed,
            "mean_delta": mean,
            "sample_std": std,
        }
    return out


def _artifact_counts(root: Path) -> dict:
    return {
        "probability_bundles": len(list(root.glob("*/lambda_*/probs/*/meta.json"))),
        "prediction_files": len(list(root.glob("*/lambda_*/predictions/*.csv.gz"))),
        "result_files": len(list(root.glob("*/lambda_*/results/*.json"))),
    }


def _input_hashes(root: Path) -> dict:
    out = {}
    for model in MODELS:
        for structure_lambda in LAMBDAS:
            arm = Path(arm_dir(
                "mlpromise_fr", model["name"], structure_lambda,
            )).relative_to("runs_fr")
            arm_root = root / arm
            for protocol in model["protocols"]:
                for seed in SEEDS:
                    for method in METHODS:
                        path = predictions_path(protocol, seed, method, arm_root)
                        out[str(path.relative_to(root))] = file_sha256(path)
    return out


def build_report(corpus: str) -> dict:
    if corpus not in LANGUAGE_CODES:
        raise ValueError(f"unsupported multilingual replication corpus {corpus!r}")
    language = LANGUAGE_CODES[corpus]
    root = REPO_ROOT / f"runs_{language}"
    order, checksum = _split_identity(corpus)

    summaries, runtimes, contrasts = {}, {}, {}
    for model in MODELS:
        slug = Path(arm_dir(corpus, model["name"], 0.0)).parts[-2]
        summaries[slug], runtimes[slug], contrasts[slug] = {}, {}, {}
        for structure_lambda in LAMBDAS:
            label = f"lambda_{structure_lambda:.1f}"
            arm_root = REPO_ROOT / arm_dir(corpus, model["name"], structure_lambda)
            by_protocol = {
                protocol: {
                    method: summarize_method(arm_root, protocol, method, order, SEEDS)
                    for method in METHODS
                }
                for protocol in model["protocols"]
            }
            summaries[slug][label] = by_protocol
            contrasts[slug][label] = {
                protocol: {
                    f"{first}-{second}": {
                        "description": description,
                        **_contrast(methods, first, second),
                    }
                    for first, second, description in CONTRASTS
                }
                for protocol, methods in by_protocol.items()
            }
            runtimes[slug][label] = _runtime(
                arm_root / "probs",
                len(model["protocols"]) * len(SEEDS) * 5,
                structure_lambda,
            )
            if runtimes[slug][label]["model_names"] != [model["name"]]:
                raise ValueError(
                    f"{slug}/{label}: model is "
                    f"{runtimes[slug][label]['model_names']}, expected {model['name']}"
                )

    pdf_arms = [
        summaries[slug][f"lambda_{structure_lambda:.1f}"]["pdf_group"]
        for slug in summaries
        for structure_lambda in LAMBDAS
    ]
    legal = [methods[method] for methods in pdf_arms for method in METHODS[1:]]
    m1_tuple_deltas = [
        methods["M1"]["tuple_accuracy_mean"] - methods["M0"]["tuple_accuracy_mean"]
        for methods in pdf_arms
    ]
    m1_f1_deltas = [
        methods["M1"]["weighted_macro_f1_mean"]
        - methods["M0"]["weighted_macro_f1_mean"]
        for methods in pdf_arms
    ]

    released = release_rows(language)
    prov = provenance(language)
    first_result = next(root.glob("*/lambda_*/results/pdf_group_seed42_M0.json"))
    with open(first_result, encoding="utf-8") as f:
        class_support = json.load(f)["per_class_support"]

    counts = _artifact_counts(root)
    expected = {
        "probability_bundles": 150,
        "prediction_files": 210,
        "result_files": 210,
    }
    if counts != expected:
        raise ValueError(f"{root}: artifact counts are {counts}, expected {expected}")

    return {
        "report_version": REPORT_VERSION,
        "design": {
            "corpus": corpus,
            "language": LANGUAGE_NAMES[language],
            "data_checksum": checksum,
            "release_sha256": prov["sha256"],
            "n_rows": len(order),
            "n_report_clusters": len({row["URL"] for row in released}),
            "input_unit": "PDF page" if language == "ko" else "paragraph",
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "lambdas": list(LAMBDAS),
            "weighted_metric_label": "AI CUP weights applied to ML-Promise (not official)",
            "class_support": class_support,
        },
        "artifact_counts": counts,
        "observations": {
            "inference": "descriptive within-corpus contrasts only",
            "pdf_group_arm_count": len(pdf_arms),
            "m0_invalid_positive_in_every_arm": all(
                methods["M0"]["invalid_tuple_rate_mean"] > 0 for methods in pdf_arms
            ),
            "m1_to_m6_max_invalid_tuple_rate": max(
                row["invalid_tuple_rate_mean"] for row in legal
            ),
            "m1_minus_m0_tuple_nonnegative_arm_count": sum(
                delta >= 0 for delta in m1_tuple_deltas
            ),
            "m1_minus_m0_weighted_f1_positive_arm_count": sum(
                delta > 0 for delta in m1_f1_deltas
            ),
        },
        "summaries": summaries,
        "contrasts": contrasts,
        "runtime": runtimes,
        "input_sha256": _input_hashes(root),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpora", nargs="+", choices=LANGUAGES,
                    default=list(LANGUAGES))
    args = ap.parse_args()

    for corpus in args.corpora:
        language = LANGUAGE_CODES[corpus]
        out = REPO_ROOT / f"runs_{language}" / "summary.json"
        report = build_report(corpus)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
            f.write("\n")
        obs = report["observations"]
        print(
            f"{corpus}: {report['artifact_counts']}; "
            f"M1-M0 tuple nonnegative in "
            f"{obs['m1_minus_m0_tuple_nonnegative_arm_count']}/"
            f"{obs['pdf_group_arm_count']} pdf_group arms -> {out}"
        )


if __name__ == "__main__":
    main()
