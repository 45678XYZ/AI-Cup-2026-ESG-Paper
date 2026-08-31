"""What an invalid tuple is made of, and what the official metric pays for it.

Two numbers the manuscript leans on had no table behind them. The Introduction
says an invalid tuple still collects 44.3% of the available weighted field
credit -- the motivating figure for the whole paper -- with no derivation
anywhere. Section 6 says 95.2% of the 1,527 invalid tuples break one of two
rules, with no breakdown. Both come from the same 1,527 rows, so they are one
table with two panels rather than two tables a page apart.

    python -m analysis.invalid_anatomy      # -> tables/table8_invalid_anatomy.tex

Every count is pooled over both protocols and all three seeds, which is the
scope the running text already claims for them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.cases import RULES, _labels, load_aligned, violation_breakdown
from analysis.load import REAL_ROOT
from analysis.metrics import FIELDS
from paper.data import REPO_ROOT, canonical_row_order, load_dev
from paper.labels import is_valid_tuple
from paper.score import FIELD_WEIGHTS
from paper.train_config import PROTOCOLS, SEEDS

TABLE_NAME = "table8_invalid_anatomy"
BASELINE = "M0"

# Prose for each rule id, in the order ``analysis.cases.RULES`` tests them.
# The order matters and is not cosmetic: violation_breakdown attributes each
# invalid row to the first rule it breaks, so a row that denies a promise and
# also answers an evidence quality is counted once, under the promise rule.
RULE_LABELS = {
    "promise_no_children_set":
        r"\textsf{PS}=No, some child not \emph{N/A}",
    "evidence_no_quality_set":
        r"\textsf{ES}=No, \textsf{EQ} substantive",
    "promise_yes_children_absent":
        r"\textsf{PS}=Yes, \textsf{VT} or \textsf{ES} \emph{N/A}",
    "evidence_yes_quality_missing":
        r"\textsf{ES}=Yes, \textsf{EQ}=\emph{N/A}",
}

FIELD_LABELS = {
    "promise_status": r"\textsf{PS}",
    "verification_timeline": r"\textsf{VT}",
    "evidence_status": r"\textsf{ES}",
    "evidence_quality": r"\textsf{EQ}",
}


def field_credit_on_invalid(gold, pred) -> dict:
    """Per-field accuracy on the rows that came out invalid.

    ``analysis.cases.partial_credit_on_invalid`` returns the weighted sum of
    exactly these accuracies. It is reported here term by term because a single
    0.443 with no derivation is the one number in the paper a reader cannot
    check, and it is the number the Introduction rests its motivation on.
    """
    invalid = np.array([not is_valid_tuple(*_labels(row)) for row in pred])
    if not invalid.any():
        return {field: 0.0 for field in FIELDS}
    return {
        field: float((gold[invalid][:, j] == pred[invalid][:, j]).mean())
        for j, field in enumerate(FIELDS)
    }


def build_report(root=REAL_ROOT, *, protocols=PROTOCOLS, seeds=SEEDS) -> dict:
    dev = load_dev()
    order = canonical_row_order(dev)

    breakdowns, credits = [], []
    for protocol in protocols:
        for seed in seeds:
            gold, pred = load_aligned(protocol, seed, BASELINE, order, root)
            breakdowns.append(violation_breakdown(pred))
            credits.append(field_credit_on_invalid(gold, pred))

    n_rows = sum(b["n_rows"] for b in breakdowns)
    n_invalid = sum(b["n_invalid"] for b in breakdowns)
    by_rule = {name: sum(b["by_rule"][name] for b in breakdowns)
               for name, _ in RULES}

    # Averaged over runs, not pooled over rows, because that is the definition
    # ``cases.py`` already writes into case_analysis.json as
    # partial_credit_on_invalid. The two differ in the fourth decimal; having
    # the paper and the JSON disagree even there is not worth the difference.
    accuracy = {field: float(np.mean([c[field] for c in credits]))
                for field in FIELDS}
    weighted = {field: FIELD_WEIGHTS[field] * accuracy[field] for field in FIELDS}

    return {
        "report_version": "1.0",
        "baseline": BASELINE,
        "protocols": list(protocols),
        "seeds": list(seeds),
        "n_rows": n_rows,
        "n_invalid": n_invalid,
        "by_rule": by_rule,
        "rule_attribution": "first matching rule, in analysis.cases.RULES order",
        "field_accuracy_on_invalid": accuracy,
        "field_weights": {f: FIELD_WEIGHTS[f] for f in FIELDS},
        "weighted_credit_by_field": weighted,
        "weighted_credit_on_invalid": float(sum(weighted.values())),
    }


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def render_table(report) -> str:
    n_invalid = report["n_invalid"]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\multicolumn{3}{l}{\emph{(a) Which implication the invalid tuple breaks}} \\",
        r"Violated implication & Rows & Share \\",
        r"\midrule",
    ]
    for name in RULE_LABELS:
        count = report["by_rule"][name]
        lines.append(f"{RULE_LABELS[name]} & {_thousands(count)} & "
                     f"{100 * count / n_invalid:.1f}\\% \\\\")
    lines += [
        r"\midrule",
        f"All invalid tuples & {_thousands(n_invalid)} & 100.0\\% \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{\emph{(b) Weighted field credit those rows still collect}} \\",
        r"Field & Weight $\times$ accuracy & Credit \\",
        r"\midrule",
    ]
    for field in FIELDS:
        lines.append(
            f"{FIELD_LABELS[field]} & "
            f"{report['field_weights'][field]:.2f} $\\times$ "
            f"{report['field_accuracy_on_invalid'][field]:.3f} & "
            f"{report['weighted_credit_by_field'][field]:.3f} \\\\")
    lines += [
        r"\midrule",
        f"Total & 1.00 & {report['weighted_credit_on_invalid']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def build_caption(report) -> str:
    n_invalid = report["n_invalid"]
    two = (report["by_rule"]["promise_no_children_set"]
           + report["by_rule"]["evidence_no_quality_set"])
    credit = report["weighted_credit_on_invalid"]
    return (
        f"Anatomy of the {_thousands(n_invalid)} invalid tuples independent "
        f"argmax (M0) emits over {_thousands(report['n_rows'])} scored rows -- "
        "both protocols, all three seeds. In (a) a row is attributed to the "
        "first implication it breaks, so the four counts partition the "
        f"{_thousands(n_invalid)} exactly. Panel (b) derives the "
        "Introduction's motivating figure: restricted to those same invalid "
        "rows, each field is scored on its own, so the row still collects its "
        f"weight times its per-field accuracy, and the four terms sum to "
        f"{credit:.3f}. That is weighted field accuracy on invalid rows, not "
        "weighted macro-F1, which has no per-row decomposition."
    )


def write_report(out_dir=None, root=REAL_ROOT) -> dict:
    out_dir = REPO_ROOT / "tables" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    with open(out_dir / "invalid_anatomy.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    (out_dir / f"{TABLE_NAME}.tex").write_text(render_table(report),
                                               encoding="utf-8")
    (out_dir / f"{TABLE_NAME}_caption.txt").write_text(
        build_caption(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--root", default=REAL_ROOT, type=Path)
    args = ap.parse_args()
    report = write_report(args.out_dir, args.root)
    print(f"anatomy -> tables/{TABLE_NAME}.tex")
    print(f"  {report['n_invalid']} invalid of {report['n_rows']} rows; "
          f"weighted credit {report['weighted_credit_on_invalid']:.4f}")


if __name__ == "__main__":
    main()
