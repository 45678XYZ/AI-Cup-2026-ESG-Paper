"""What enforcing legality costs, measured identically in every arm.

The paper's central claim is that projecting onto the 17 legal states is free.
That is a contrast *within* an arm -- M1 against M0 -- and no existing artifact
reports it. ``structural_arm`` and ``architecture_screen`` both hold the method
fixed and vary the arm, which answers a different question: what the structural
loss does. Reading the projection's cost off those files means subtracting
numbers that were never paired, across four documents.

So this module computes one row per (backbone, lambda) arm, on the primary
``pdf_group`` protocol, with the study's own paired PDF-cluster bootstrap. The
clusters are shared across arms, which is what lets the rows be compared even
though they come from different training runs: within a row the pairing is
exact, and that is the only pairing the claim needs.

Two things are worth stating about what the table can and cannot say. M1's
invalid rate is zero by construction in every arm -- its output space *is* the
legal set -- so that column is a check that the pipeline did what it claims,
not a result. And the seven rows are not a Holm family: the contrast was named
after the primary analysis, so `docs/inference_families.md` classifies it as
exploratory, and the claim rests on the sign pattern across arms rather than on
any single cell's p-value.

    python -m analysis.legality_cost            # -> tables/legality_cost.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT, paired_delta
from analysis.load import load_aligned, pdf_clusters, predictions_path
from analysis.metrics import tuple_accuracy, weighted_macro_f1
from paper.data import REPO_ROOT, canonical_row_order, file_sha256, load_dev
from paper.labels import EVAL_FIELDS, FIELDS, INVALID_STATE_ID, tuple_to_state_id
from paper.provenance import environment, git_sha, now_iso
from paper.train_config import SEEDS

# The contrast is M0 against M1: the unconstrained argmax against the same
# probabilities projected onto the legal set. Every other method differs from
# M0 in more than one respect.
BASELINE, CONSTRAINED = "M0", "M1"

# Only pdf_group: the two architecture screens were pre-registered on it alone,
# and a table whose rows cover different protocols would not be comparable.
PROTOCOL = "pdf_group"

METRICS = {
    "official_weighted_macro_f1": weighted_macro_f1,
    "tuple_accuracy": tuple_accuracy,
}


@dataclass(frozen=True)
class Arm:
    """One training configuration whose predictions exist on disk.

    ``root`` is relative to the repository; the frozen anchor lives at the top
    level and is the empty string. ``registration`` names the document that
    declared the arm before it ran, which is what `inference_families.md`
    classifies by.
    """

    backbone: str
    structure_lambda: float
    root: str
    registration: str

    @property
    def label(self) -> str:
        return f"{self.backbone} (lambda={self.structure_lambda:g})"


ARMS = (
    Arm("RoBERTa-large", 0.0, "", "paper_plan.md"),
    Arm("RoBERTa-large", 0.3, "structural_arm",
        "pre_registration_structural_training.md"),
    Arm("DeBERTa-v2-320M", 0.0, "architecture_screen/deberta_v2_320m/lambda_0.0",
        "pre_registration_deberta_screen.md"),
    Arm("DeBERTa-v2-320M", 0.3, "architecture_screen/deberta_v2_320m/lambda_0.3",
        "pre_registration_deberta_screen.md"),
    Arm("ELECTRA-large", 0.0, "architecture_screen/electra_180g_large/lambda_0.0",
        "pre_registration_electra_screen.md"),
    Arm("ELECTRA-large", 0.3, "architecture_screen/electra_180g_large/lambda_0.3",
        "pre_registration_electra_screen.md"),
    Arm("RoBERTa-base", 0.0, "runs/rbt_base", "rbt_base_run.md"),
)


def _state_ids(codes) -> np.ndarray:
    names = [[EVAL_FIELDS[f][c] for f, c in zip(FIELDS, row)] for row in codes]
    return np.array([tuple_to_state_id(*n) for n in names], dtype=np.int64)


def _invalid_rate(pair) -> float:
    """Share of rows whose predicted tuple is none of the 17 legal states."""
    return float((_state_ids(pair[1]) == INVALID_STATE_ID).mean())


def arm_root(arm, root=REPO_ROOT) -> Path:
    return Path(root) / arm.root if arm.root else Path(root)


def arm_legality_cost(root, order, dev=None, *, protocol=PROTOCOL, seeds=SEEDS,
                      clusters=None, n_boot=N_BOOT,
                      bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    """M1 against M0 for one arm: invalid rates, and the cost on each metric."""
    dev = dev if dev is not None else load_dev()
    clusters = clusters if clusters is not None else pdf_clusters(order, dev)

    baseline = [load_aligned(protocol, s, BASELINE, order, root) for s in seeds]
    constrained = [load_aligned(protocol, s, CONSTRAINED, order, root) for s in seeds]

    out = {
        "protocol": protocol,
        "seeds": list(seeds),
        "invalid_rate": {
            BASELINE: float(np.mean([_invalid_rate(p) for p in baseline])),
            CONSTRAINED: float(np.mean([_invalid_rate(p) for p in constrained])),
        },
    }
    for name, score in METRICS.items():
        out[name] = {
            **paired_delta(constrained, baseline, clusters, n_boot=n_boot,
                           seed=bootstrap_seed, score=score),
            "contrast": f"{CONSTRAINED}-{BASELINE}",
            "baseline_mean": float(np.mean([score(*p) for p in baseline])),
            "constrained_mean": float(np.mean([score(*p) for p in constrained])),
        }
    return out


def _input_hashes(root, protocol, seeds) -> dict:
    """sha256 of every prediction file the row was computed from."""
    out = {}
    for seed in seeds:
        for method in (BASELINE, CONSTRAINED):
            path = predictions_path(protocol, seed, method, root)
            out[f"{protocol}_seed{seed}_{method}.csv.gz"] = file_sha256(path)
    return out


def build_report(root=REPO_ROOT, arms=ARMS, *, protocol=PROTOCOL, seeds=SEEDS,
                 n_boot=N_BOOT, bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    dev = load_dev()
    order = canonical_row_order(dev)
    clusters = pdf_clusters(order, dev)

    rows = []
    for arm in arms:
        this = arm_root(arm, root)
        rows.append({
            "label": arm.label,
            "backbone": arm.backbone,
            "structure_lambda": arm.structure_lambda,
            "root": arm.root,
            "registration": arm.registration,
            **arm_legality_cost(this, order, dev, protocol=protocol, seeds=seeds,
                                clusters=clusters, n_boot=n_boot,
                                bootstrap_seed=bootstrap_seed),
            "input_sha256": _input_hashes(this, protocol, seeds),
        })

    return {
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "environment": environment(),
        "contrast": f"{CONSTRAINED} - {BASELINE}",
        "protocol": protocol,
        "n_boot": n_boot,
        "bootstrap_seed": bootstrap_seed,
        "inference_family": "exploratory (docs/inference_families.md)",
        "arms": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    out = args.out or Path(args.root) / "tables" / "legality_cost.json"
    report = build_report(args.root, n_boot=args.n_boot)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"legality cost -> {out}")


if __name__ == "__main__":
    main()
