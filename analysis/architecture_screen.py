"""Compare one exploratory backbone's lambda=0 and structural-loss arms.

The structural contrast stays paired across the same rows, PDF clusters and
seeds. A second, separately labelled contrast compares the backbone's lambda=0
arm with the frozen RoBERTa lambda=0 anchor; that is a total backbone difference,
not evidence about the structural loss.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT, paired_delta
from analysis.load import METHODS, load_aligned, pdf_clusters, predictions_path
from analysis.structural_arm import NAMED_SAFETY_CLASSES, _protocol_comparison
from paper.data import REPO_ROOT, canonical_row_order, file_sha256, load_dev
from paper.train_config import SEEDS

PROTOCOL = "pdf_group"
H1_METHOD = "M0"
H2_METHOD = "M1"


def _load_sets(root, method, order, seeds):
    return [load_aligned(PROTOCOL, seed, method, order, root) for seed in seeds]


def _assert_same_gold(sets_a, sets_b, label):
    for (gold_a, _), (gold_b, _) in zip(sets_a, sets_b):
        if not np.array_equal(gold_a, gold_b):
            raise ValueError(f"{label}: arms disagree on aligned gold")


def compare_architecture(baseline_root, structural_root, anchor_root, *,
                         n_boot=N_BOOT, bootstrap_seed=BOOTSTRAP_SEED,
                         seeds=SEEDS, dev=None) -> dict:
    """Return gate, structural-effect and backbone-total-difference results."""
    dev = dev if dev is not None else load_dev()
    order = canonical_row_order(dev)
    clusters = pdf_clusters(order, dev)

    protocol = _protocol_comparison(
        PROTOCOL, order, baseline_root, structural_root, seeds,
    )
    baseline_m1, structural_m1 = protocol.pop("_m1_sets")
    structural_effect = paired_delta(
        structural_m1, baseline_m1, clusters,
        n_boot=n_boot, seed=bootstrap_seed,
    )
    structural_effect.update({
        "protocol": PROTOCOL,
        "contrast": "architecture lambda=0.3 M1 - architecture lambda=0 M1",
        "alpha": 0.05,
        "multiplicity_adjustment": None,
        "exploratory": True,
        "supported": structural_effect["delta"] > 0
        and structural_effect["p_value"] < 0.05,
    })

    anchor_m1 = _load_sets(anchor_root, H2_METHOD, order, seeds)
    _assert_same_gold(baseline_m1, anchor_m1, "backbone comparison")
    backbone_difference = paired_delta(
        baseline_m1, anchor_m1, clusters,
        n_boot=n_boot, seed=bootstrap_seed,
    )
    backbone_difference.update({
        "protocol": PROTOCOL,
        "contrast": "architecture lambda=0 M1 - frozen RoBERTa lambda=0 M1",
        "interpretation": "total_backbone_difference_not_structural_effect",
        "exploratory": True,
    })

    invalid = protocol["h1_invalid_tuple_rate"]
    score = protocol["h2_weighted_macro_f1"]
    gate = {
        "evaluated_on_seed": int(seeds[0]),
        "invalid_rate_lower": invalid["structural_per_seed"][0]
        < invalid["baseline_per_seed"][0],
        "m1_weighted_macro_f1_higher": score["structural_per_seed"][0]
        > score["baseline_per_seed"][0],
    }
    gate["passed"] = gate["invalid_rate_lower"] and gate["m1_weighted_macro_f1_higher"]

    named = {}
    classes = protocol["m1_per_class_f1"]
    for field, label in NAMED_SAFETY_CLASSES:
        named[f"{field}:{label}"] = classes[field][label]

    return {
        "design": {
            "protocol": PROTOCOL,
            "baseline": "architecture lambda=0",
            "structural": "architecture lambda=0.3",
            "anchor": "frozen RoBERTa lambda=0",
            "seeds": list(seeds),
            "h1_method": H1_METHOD,
            "h2_method": H2_METHOD,
            "bootstrap_unit": "pdf_url",
            "bootstrap_seed": bootstrap_seed,
            "conditional_expansion": True,
        },
        "seed42_expansion_gate": gate,
        "protocols": {PROTOCOL: protocol},
        "structural_effect_m1": structural_effect,
        "backbone_total_difference_m1": backbone_difference,
        "safety_check": {
            "protocol": PROTOCOL,
            "method": H2_METHOD,
            "named_rare_classes": named,
            "named_rare_class_decline": any(row["delta"] < 0 for row in named.values()),
        },
    }


def _bundle_meta(probs_dir: Path, expected_lambda: float) -> dict:
    paths = sorted(Path(probs_dir).glob("*/meta.json"))
    if not paths:
        raise ValueError(f"no probability bundles under {probs_dir}")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    keys = ("model_name", "model_revision", "git_sha", "hardware", "structure_lambda")
    values = {key: {str(row.get(key)) for row in rows} for key in keys}
    mixed = {key: sorted(value) for key, value in values.items() if len(value) != 1}
    if mixed:
        raise ValueError(f"bundles mix runtime metadata: {mixed}")
    actual_lambda = float(next(iter(values["structure_lambda"])))
    if actual_lambda != expected_lambda:
        raise ValueError(
            f"{probs_dir} has structure_lambda={actual_lambda}, expected {expected_lambda}"
        )
    return {
        "bundle_count": len(rows),
        "model_name": next(iter(values["model_name"])),
        "model_revision": next(iter(values["model_revision"])),
        "git_sha": next(iter(values["git_sha"])),
        "hardware": next(iter(values["hardware"])),
        "structure_lambda": actual_lambda,
        "total_train_seconds": float(sum(row["train_seconds"] for row in rows)),
    }


def _input_hashes(baseline_root, structural_root, anchor_root, seeds) -> dict:
    paths = {}
    for arm, root in (("baseline", baseline_root), ("structural", structural_root)):
        for seed in seeds:
            for method in METHODS:
                path = predictions_path(PROTOCOL, seed, method, root)
                paths[f"{arm}/{path.name}"] = file_sha256(path)
    for seed in seeds:
        path = predictions_path(PROTOCOL, seed, H2_METHOD, anchor_root)
        paths[f"anchor/{path.name}"] = file_sha256(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--structural-root", type=Path, required=True)
    ap.add_argument("--anchor-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--baseline-probs-dir", type=Path, required=True)
    ap.add_argument("--structural-probs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    report = compare_architecture(
        args.baseline_root, args.structural_root, args.anchor_root,
        n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed, seeds=args.seeds,
    )
    report["runtime"] = {
        "baseline": _bundle_meta(args.baseline_probs_dir, 0.0),
        "structural": _bundle_meta(args.structural_probs_dir, 0.3),
    }
    report["input_sha256"] = _input_hashes(
        args.baseline_root, args.structural_root, args.anchor_root, args.seeds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")

    effect = report["structural_effect_m1"]
    invalid = report["protocols"][PROTOCOL]["h1_invalid_tuple_rate"]
    print(f"seed42 expansion gate passed: {report['seed42_expansion_gate']['passed']}")
    print(
        f"M0 invalid {invalid['baseline_mean']:.3%} -> "
        f"{invalid['structural_mean']:.3%}"
    )
    print(
        f"M1 delta={effect['delta']:.6f} "
        f"95% CI [{effect['ci_low']:.6f}, {effect['ci_high']:.6f}] "
        f"p={effect['p_value']:.4f} supported={effect['supported']}"
    )
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
