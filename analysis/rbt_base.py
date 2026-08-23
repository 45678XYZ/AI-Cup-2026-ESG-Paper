"""Summarize the pre-registered RBT-base backbone generality check."""

import argparse
import json
from pathlib import Path

from paper.data import REPO_ROOT, file_sha256
from paper.train_config import SEEDS

PROTOCOLS = ("pdf_group", "row_strat")


def _result(root: Path, protocol: str, seed: int, method: str) -> dict:
    path = Path(root) / "results" / f"{protocol}_seed{seed}_{method}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(values):
    return sum(values) / len(values)


def summarize_protocol(run_root: Path, anchor_root: Path, protocol: str,
                       seeds=SEEDS) -> dict:
    """Compare M0 invalid and M1-M0 for one protocol using exact result JSON."""
    architecture_m0 = [_result(run_root, protocol, seed, "M0") for seed in seeds]
    architecture_m1 = [_result(run_root, protocol, seed, "M1") for seed in seeds]
    anchor_m0 = [_result(anchor_root, protocol, seed, "M0") for seed in seeds]
    anchor_m1 = [_result(anchor_root, protocol, seed, "M1") for seed in seeds]

    def metrics(m0, m1):
        invalid = [row["invalid_tuple_rate"] for row in m0]
        m0_score = [row["weighted_macro_f1"] for row in m0]
        m1_score = [row["weighted_macro_f1"] for row in m1]
        contrast = [b - a for a, b in zip(m0_score, m1_score)]
        return {
            "m0_invalid_per_seed": invalid,
            "m0_invalid_mean": _mean(invalid),
            "m0_weighted_macro_f1_per_seed": m0_score,
            "m0_weighted_macro_f1_mean": _mean(m0_score),
            "m1_weighted_macro_f1_per_seed": m1_score,
            "m1_weighted_macro_f1_mean": _mean(m1_score),
            "m1_minus_m0_per_seed": contrast,
            "m1_minus_m0_mean": _mean(contrast),
        }

    architecture = metrics(architecture_m0, architecture_m1)
    anchor = metrics(anchor_m0, anchor_m1)
    hypothesis = {
        "architecture_invalid_higher_than_anchor":
            architecture["m0_invalid_mean"] > anchor["m0_invalid_mean"],
        "architecture_m1_minus_m0_higher_than_anchor":
            architecture["m1_minus_m0_mean"] > anchor["m1_minus_m0_mean"],
    }
    hypothesis["both_conditions_met"] = all(hypothesis.values())

    return {
        "architecture": architecture,
        "frozen_large_anchor": anchor,
        "differences": {
            "m0_invalid_mean": architecture["m0_invalid_mean"]
            - anchor["m0_invalid_mean"],
            "m1_minus_m0_mean": architecture["m1_minus_m0_mean"]
            - anchor["m1_minus_m0_mean"],
            "m1_absolute_mean": architecture["m1_weighted_macro_f1_mean"]
            - anchor["m1_weighted_macro_f1_mean"],
        },
        "generality_hypothesis": hypothesis,
    }


def _runtime(probs_dir: Path) -> dict:
    paths = sorted(Path(probs_dir).glob("*/meta.json"))
    if not paths:
        raise ValueError(f"no bundles under {probs_dir}")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    keys = ("model_name", "model_revision", "git_sha", "hardware")
    values = {key: {str(row.get(key)) for row in rows} for key in keys}
    mixed = {key: sorted(value) for key, value in values.items() if len(value) != 1}
    if mixed:
        raise ValueError(f"bundles mix runtime metadata: {mixed}")
    return {
        "bundle_count": len(rows),
        **{key: next(iter(value)) for key, value in values.items()},
        "total_train_seconds": sum(row["train_seconds"] for row in rows),
    }


def _input_hashes(run_root: Path, anchor_root: Path, seeds=SEEDS) -> dict:
    out = {}
    for label, root in (("architecture", run_root), ("anchor", anchor_root)):
        for protocol in PROTOCOLS:
            for seed in seeds:
                for method in ("M0", "M1"):
                    path = Path(root) / "results" / f"{protocol}_seed{seed}_{method}.json"
                    out[f"{label}/{path.name}"] = file_sha256(path)
    return out


def build_report(run_root: Path, anchor_root: Path, seeds=SEEDS) -> dict:
    return {
        "design": {
            "question": "does a smaller same-family backbone increase invalid output and M1-M0",
            "protocols": list(PROTOCOLS),
            "seeds": list(seeds),
            "methods": ["M0", "M1"],
            "inference": "descriptive_exploratory",
        },
        "protocols": {
            protocol: summarize_protocol(run_root, anchor_root, protocol, seeds)
            for protocol in PROTOCOLS
        },
        "runtime": _runtime(Path(run_root) / "probs"),
        "input_sha256": _input_hashes(run_root, anchor_root, seeds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=Path("runs/rbt_base"))
    ap.add_argument("--anchor-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path, default=Path("runs/rbt_base/comparison.json"))
    args = ap.parse_args()

    report = build_report(args.run_root, args.anchor_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")

    pdf = report["protocols"]["pdf_group"]
    print(f"pdf invalid base={pdf['architecture']['m0_invalid_mean']:.3%} "
          f"anchor={pdf['frozen_large_anchor']['m0_invalid_mean']:.3%}")
    print(f"pdf M1-M0 base={pdf['architecture']['m1_minus_m0_mean']:+.6f} "
          f"anchor={pdf['frozen_large_anchor']['m1_minus_m0_mean']:+.6f}")
    print(f"hypothesis met: {pdf['generality_hypothesis']['both_conditions_met']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
