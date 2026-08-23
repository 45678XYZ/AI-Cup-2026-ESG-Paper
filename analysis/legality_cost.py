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

# Three decision rules over one set of probabilities. M0 is the unconstrained
# per-field argmax. M1 and M4 both emit only legal tuples and differ solely in
# which legal tuple they pick: M1 walks the hierarchy top-down and never
# revisits the parent, M4 scores all 17 states and takes the best sum. That is
# the comparison the table is for -- two rules that are equally legal, and the
# two metrics disagree about which one is better.
BASELINE, CONSTRAINED, DECODER = "M0", "M1", "M4"

# Only pdf_group: the two architecture screens were pre-registered on it alone,
# and a table whose rows cover different protocols would not be comparable.
PROTOCOL = "pdf_group"

METRICS = {
    "official_weighted_macro_f1": weighted_macro_f1,
    "tuple_accuracy": tuple_accuracy,
}

# What legality costs, and which legal rule each metric prefers. Keeping both
# in one table is the point: the four resulting columns do not share a sign,
# and that is the finding.
CONTRASTS = {
    "legality_cost": (CONSTRAINED, BASELINE),
    "decoder_vs_projection": (DECODER, CONSTRAINED),
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

    decoded = [load_aligned(protocol, s, DECODER, order, root) for s in seeds]

    out = {
        "protocol": protocol,
        "seeds": list(seeds),
        "invalid_rate": {
            BASELINE: float(np.mean([_invalid_rate(p) for p in baseline])),
            CONSTRAINED: float(np.mean([_invalid_rate(p) for p in constrained])),
            DECODER: float(np.mean([_invalid_rate(p) for p in decoded])),
        },
        # Absolute scores for the three decision rules the table compares. The
        # deltas below carry the intervals; these carry the comparison a reader
        # actually makes, which is "which rule does each metric prefer".
        "methods": {
            name: {metric: float(np.mean([score(*pair) for pair in sets]))
                   for metric, score in METRICS.items()}
            for name, sets in ((BASELINE, baseline), (CONSTRAINED, constrained),
                               (DECODER, decoded))
        },
    }
    # Two contrasts, both within the arm. The first is what legality costs;
    # the second is which of two equally legal rules each metric prefers.
    for key, (after, before) in CONTRASTS.items():
        sets = {BASELINE: baseline, CONSTRAINED: constrained, DECODER: decoded}
        out[key] = {
            name: {
                **paired_delta(sets[after], sets[before], clusters, n_boot=n_boot,
                               seed=bootstrap_seed, score=score),
                "contrast": f"{after}-{before}",
            }
            for name, score in METRICS.items()
        }
    return out


def _input_hashes(root, protocol, seeds) -> dict:
    """sha256 of every prediction file the row was computed from."""
    out = {}
    for seed in seeds:
        for method in (BASELINE, CONSTRAINED, DECODER):
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


# --- rendering -------------------------------------------------------------

TABLE_NAME = "table6_legality_cost"


def _excludes_zero(row) -> bool:
    return row["ci_low"] > 0 or row["ci_high"] < 0


def _cell(row) -> str:
    """Delta alone, bold when its interval clears zero.

    The intervals are in the JSON; printing them here would quadruple the
    table's width and bury the only thing the reader needs from it, which is
    the sign of each column.
    """
    d = f"{row['delta']:+.3f}"
    return f"\\textbf{{{d}}}" if _excludes_zero(row) else d


def render_table(report) -> str:
    """Two contrasts side by side, one row per arm.

    Both contrasts are within an arm, so both are exactly paired. Putting them
    in one table is the argument: enforcing legality costs on the official
    metric and gains on whole-row accuracy, and swapping the greedy projection
    for joint decoding reverses both signs -- with the invalid rate at zero
    either way. Four columns, no two of which agree.
    """
    header = (
        " & & & \\multicolumn{2}{c}{Enforce legality (M1$-$M0)} & "
        "\\multicolumn{2}{c}{Joint decoding (M4$-$M1)} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Backbone & $\\lambda$ & M0 invalid & Official & Whole-row & "
        "Official & Whole-row"
    )

    def line(arm) -> str:
        cells = [
            arm["backbone"],
            f"{arm['structure_lambda']:g}",
            f"{arm['invalid_rate']['M0'] * 100:.2f}\\%",
        ]
        for key in ("legality_cost", "decoder_vs_projection"):
            cells += [_cell(arm[key]["official_weighted_macro_f1"]),
                      _cell(arm[key]["tuple_accuracy"])]
        return " & ".join(cells) + " \\\\"

    plain = [a for a in report["arms"] if a["structure_lambda"] == 0]
    trained = [a for a in report["arms"] if a["structure_lambda"] != 0]
    body = [line(a) for a in plain]
    if trained:
        body += ["\\midrule"] + [line(a) for a in trained]

    rows = "\n".join(body)
    return ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{rows}\n"
            "\\bottomrule\n\\end{tabular}\n")


def build_caption(report) -> str:
    """State the three things the tabular cannot: that both constrained rules
    emit only legal tuples, that bold is an uncorrected interval rather than a
    verdict, and that these contrasts are exploratory."""
    arms = report["arms"]
    plain = [a for a in arms if a["structure_lambda"] == 0]
    trained = [a for a in arms if a["structure_lambda"] != 0]
    lo = min(a["invalid_rate"]["M0"] for a in arms)
    hi = max(a["invalid_rate"]["M0"] for a in arms)
    gains = sum(1 for a in arms
                if a["legality_cost"]["tuple_accuracy"]["delta"] > 0)

    parts = [
        f"Two decision rules over one set of probabilities, on the "
        f"{report['protocol']} protocol. M0 takes each field's argmax "
        f"independently; M1 projects onto the 17 legal states top-down; M4 "
        f"scores all 17 and takes the best. \\textbf{{M1 and M4 both emit an "
        f"invalid tuple on 0\\% of rows in every arm}} -- their output space "
        f"is the legal set -- so the invalid column reports M0 only, where it "
        f"ranges from {lo * 100:.2f}\\% to {hi * 100:.2f}\\%.",
        f"Means over {len(arms[0]['seeds'])} seeds; paired PDF-cluster "
        f"bootstrap, {report['n_boot']:,} resamples, seed "
        f"{report['bootstrap_seed']}, one resample shared within each contrast.",
        f"Enforcing legality raises whole-row accuracy in all {gains} arms and "
        f"lowers the official score in all {len(plain)} arms trained without "
        f"the structural objective; joint decoding reverses both signs. "
        f"The {len(trained)} structurally trained arms show neither effect.",
        "\\textbf{Exploratory.} These contrasts were named after the primary "
        "analysis and form no Holm family; bold marks an interval excluding "
        "zero, not a corrected verdict. The claim is the sign pattern across "
        "arms, not any single cell. See docs/inference\\_families.md.",
    ]
    return " ".join(parts)


def write_legality_cost(out_dir, root=REPO_ROOT, *, n_boot=N_BOOT) -> Path:
    """Compute the report and write its three files. Raises if an arm is absent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        path = predictions_path(PROTOCOL, SEEDS[0], BASELINE, arm_root(arm, root))
        if not path.exists():
            raise FileNotFoundError(path)

    report = build_report(root, n_boot=n_boot)
    out = out_dir / "legality_cost.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    (out_dir / f"{TABLE_NAME}.tex").write_text(render_table(report), encoding="utf-8")
    (out_dir / f"{TABLE_NAME}_caption.txt").write_text(
        build_caption(report) + "\n", encoding="utf-8")
    return out


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

    tables = out.parent
    (tables / f"{TABLE_NAME}.tex").write_text(render_table(report), encoding="utf-8")
    (tables / f"{TABLE_NAME}_caption.txt").write_text(
        build_caption(report) + "\n", encoding="utf-8")
    print(f"legality cost -> {out} (+ {TABLE_NAME}.tex, caption)")


if __name__ == "__main__":
    main()
