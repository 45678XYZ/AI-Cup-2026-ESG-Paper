"""The 32 external arms behind the 32/32 and 14/32 summaries, one row per arm.

Those two fractions are the most-repeated numbers in the manuscript -- abstract,
sections 6, 7 and 8 -- and until now the only external table showed five fixed
arms. A reader who wanted to check 14/32 had no way to. This enumerates every
executed arm: four corpora, four backbones each, two training objectives.

    python -m analysis.external_arms       # -> tables/table9_external_arms.tex

The numbers are read from the per-language ``runs_*/summary.json`` written by
the replication runs, never recomputed here, so this table cannot drift from
the artifacts the replication itself reported.
"""

import argparse
import json
import statistics
from pathlib import Path

from paper.data import REPO_ROOT

TABLE_NAME = "table9_external_arms"

# (directory, printed corpus name). English keeps a different summary schema
# from the other three because its replication was written first; both shapes
# are read rather than one being rewritten, since the summaries are frozen
# artifacts of runs that are not being re-executed.
CORPORA = (
    ("runs_en", "English"),
    ("runs_fr", "French"),
    ("runs_ja", "Japanese"),
    ("runs_ko", "Korean"),
)

# Checkpoint directory to printed name. Every name here is prefixed by its
# corpus in the table's first column; the Chinese anchor's screens live in
# table 6 and are prefixed ``Chinese`` there, because ``RoBERTa-large`` alone
# names three different checkpoints across this paper.
BACKBONES = {
    "roberta_large": "RoBERTa-large",
    "deberta_v3_large": "DeBERTa-v3-large",
    "electra_large_discriminator": "ELECTRA-large",
    "roberta_base": "RoBERTa-base",
    "xlm_roberta_large": "XLM-R-large",
    "xlm_roberta_base": "XLM-R-base",
    "rembert": "RemBERT",
    "bert_base_multilingual_cased": "mBERT",
}

# The order each language's four backbones are printed in: primary fixed arm
# first, so the row the multilingual table already reports is the row a reader
# lands on first.
ORDER = {
    "runs_en": ("roberta_large", "deberta_v3_large",
                "electra_large_discriminator", "roberta_base"),
}
DEFAULT_ORDER = ("xlm_roberta_large", "rembert", "xlm_roberta_base",
                 "bert_base_multilingual_cased")

LAMBDAS = ("lambda_0.0", "lambda_0.3")


def _english_arms(summary) -> dict:
    """English stores its contrasts under the pre-registered hypothesis id."""
    per_backbone = summary["hypotheses"]["H-EN4"]["per_backbone"]
    return {
        backbone: {
            lam: {
                "invalid": entry["m0_invalid_tuple_rate"],
                "d_wf1": entry["m1_minus_m0_weighted_macro_f1"],
                "d_tuple": entry["m1_minus_m0_tuple_accuracy"],
            }
            for lam, entry in lams.items()
        }
        for backbone, lams in per_backbone.items()
    }


def _standard_arms(summary) -> dict:
    """French, Japanese and Korean keep contrasts and summaries side by side."""
    out = {}
    for backbone, lams in summary["contrasts"].items():
        out[backbone] = {}
        for lam, protocols in lams.items():
            contrast = protocols["pdf_group"]["M1-M0"]
            m0 = summary["summaries"][backbone][lam]["pdf_group"]["M0"]
            out[backbone][lam] = {
                "invalid": statistics.mean(
                    seed["invalid_tuple_rate"] for seed in m0["per_seed"]),
                "d_wf1": contrast["weighted_macro_f1"]["mean_delta"],
                "d_tuple": contrast["tuple_accuracy"]["mean_delta"],
            }
    return out


def build_report(repo_root=REPO_ROOT) -> dict:
    repo_root = Path(repo_root)
    corpora = []
    for directory, name in CORPORA:
        path = repo_root / directory / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        arms = (_english_arms(summary) if "hypotheses" in summary
                else _standard_arms(summary))
        corpora.append({
            "corpus": name,
            "directory": directory,
            "order": list(ORDER.get(directory, DEFAULT_ORDER)),
            "arms": arms,
        })

    flat = [entry
            for corpus in corpora
            for backbone in corpus["order"]
            for lam in LAMBDAS
            for entry in [corpus["arms"][backbone][lam]]]
    return {
        "report_version": "1.0",
        "contrast": "M1-M0, document-disjoint, mean over three seeds",
        "corpora": corpora,
        "n_arms": len(flat),
        "n_tuple_positive": sum(1 for e in flat if e["d_tuple"] > 0),
        "n_weighted_positive": sum(1 for e in flat if e["d_wf1"] > 0),
        "weighted_positive_by_corpus": {
            corpus["corpus"]: sum(
                1 for backbone in corpus["order"] for lam in LAMBDAS
                if corpus["arms"][backbone][lam]["d_wf1"] > 0)
            for corpus in corpora
        },
    }


def render_table(report) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$\lambda=0$} & "
        r"\multicolumn{3}{c}{$\lambda=0.3$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Corpus & Backbone & M0 inv.\ \% & $\Delta$wF1 & $\Delta$tuple & "
        r"M0 inv.\ \% & $\Delta$wF1 & $\Delta$tuple \\",
    ]
    for corpus in report["corpora"]:
        lines.append(r"\midrule")
        for i, backbone in enumerate(corpus["order"]):
            cells = ["\\emph{%s}" % corpus["corpus"] if i == 0 else "",
                     BACKBONES[backbone]]
            for lam in LAMBDAS:
                entry = corpus["arms"][backbone][lam]
                cells += [f"{entry['invalid'] * 100:.2f}",
                          f"{entry['d_wf1']:+.4f}",
                          f"\\textbf{{{entry['d_tuple']:+.4f}}}"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_caption(report) -> str:
    """Only what the table cannot say for itself.

    The counts, their per-corpus split and the not-a-ranking caveat are all in
    the Results text already; repeating them here cost most of a column on an
    eight-page limit. What survives is what a reader cannot reconstruct from
    the cells: what one cell is, why four decimals, and whose checkpoints
    these are.
    """
    return (
        "Every executed external arm behind the "
        f"{report['n_tuple_positive']}/{report['n_arms']} and "
        f"{report['n_weighted_positive']}/{report['n_arms']} summaries. Each "
        "cell is the M1$-$M0 within-arm contrast under document-disjoint "
        "evaluation, averaged over three seeds; the arms were fixed before any "
        "score was inspected. Bold marks a positive tuple-accuracy contrast. "
        "Four decimals, because three would print several arms as "
        "$\\pm$0.000 and hide the sign the counts are made of. "
        # \\ref rather than a typed number: this caption is generated, and
        # inserting the evidence tables renumbers every float after them.
        "Backbones are corpus-prefixed; the identically named checkpoints in "
        "Table~\\ref{tab:legality-cost} are Chinese. ML-Promise has no "
        "official weighted metric, so its wF1 applies AI CUP weights."
    )


def write_report(out_dir=None, repo_root=REPO_ROOT) -> dict:
    out_dir = REPO_ROOT / "tables" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(repo_root)
    with open(out_dir / "external_arms.json", "w", encoding="utf-8") as f:
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
    args = ap.parse_args()
    report = write_report(args.out_dir)
    print(f"arms    -> tables/{TABLE_NAME}.tex")
    print(f"  tuple positive {report['n_tuple_positive']}/{report['n_arms']}, "
          f"weighted positive {report['n_weighted_positive']}/{report['n_arms']}")


if __name__ == "__main__":
    main()
