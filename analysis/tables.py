r"""Emit contract-4 tables: tabular only, captions beside them, manifest on top.

Contract section 5 splits the responsibility deliberately: content is C's
(number correctness), layout is D's (the 8-page budget). So nothing here emits
``\begin{table}``, ``\caption`` or ``\label``, and only booktabs rules are
used. Column counts follow ``contracts/examples/tables/`` because that is what
D measured the page budget against.

Every cell is computed. The manifest records the sha256 of every input file so
any number in the paper can be traced back to the artifacts that produced it
(contract section 5).
"""

import json
from pathlib import Path

from paper.data import file_sha256
from paper.labels import FIELDS
from paper.provenance import git_sha, now_iso

CONTRACT_VERSION = "1.0"

TABLE_FILES = ("table1_dataset.tex", "table2_main.tex", "table3_regimes.tex")

# The competition test split ships no labels, so its label-derived cells cannot
# be filled. Printing a development number there would be a fabricated cell.
NA = "n/a"

TABLE2_ROWS = (
    ("M0", "None", "Independent"), ("M1", "None", "Projection"),
    ("M2", "Global", "Projection"), ("M3", "Conditional", "Projection"),
    ("M4", "None", "17-state"), ("M5", "Global", "17-state"),
    ("M6", "Conditional", "17-state"),
)

CAPTIONS = {
    "table1_dataset": (
        "Dataset and split statistics, regenerated from the versioned data by "
        "analysis/audit.py. Misleading occurs twice in the whole development "
        "set, in two different reports, and is absent from the Calibration "
        "partition of 18 of the 30 rotations. The competition test split ships "
        "no labels, so its label-derived cells are marked n/a."
    ),
    "table2_main": (
        "Controlled comparison of decision rules on identical base "
        "probabilities and identical test rows. Each cell is the mean over "
        "three seeds of a single score computed on all 2,000 concatenated test "
        "rows; $\\pm$ is the sample standard deviation across seeds, which "
        "reflects the variability of the whole pipeline -- fold assignment and "
        "training together -- rather than model stability."
    ),
    "table3_regimes": (
        "Same-document versus document-disjoint evaluation. The left column "
        "measures seen-report, unseen-paragraph generalisation and matches the "
        "competition distribution, since test and development draw on the same "
        "49 reports; the right column measures generalisation to entirely "
        "unseen reports. $\\Delta$ is the gap between two estimation targets "
        "and is not a bias estimate."
    ),
}


def _f(value, places=3):
    return f"{value:.{places}f}"


def _pm(mean, std):
    return f"{_f(mean)}$\\pm${_f(std)}"


def _tabular(spec, header, body_lines):
    rows = "\n".join(body_lines)
    return (
        f"\\begin{{tabular}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_table1(audit) -> str:
    dev, test = audit["development"], audit["test"]

    def row(label, dev_value, test_value):
        return f"{label} & {dev_value} & {test_value} \\\\"

    eq = dev["class_support"]["evidence_quality"]
    vt = dev["class_support"]["verification_timeline"]
    body = [
        row("Paragraphs", dev["paragraphs"], test["paragraphs"]),
        row("Source reports (PDFs)", dev["pdfs"], test["pdfs"]),
        row("Companies", dev["companies"], test["companies"]),
        row("Legal states observed", f"{dev['legal_states_observed']} / 17", NA),
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Rarest classes}} \\\\",
        row("\\quad within\\_2\\_years", vt["within_2_years"], NA),
        row("\\quad Misleading", eq["Misleading"], NA),
    ]
    return _tabular("lrr", "Statistic & Development & Test", body)


def render_table2(summary) -> str:
    body = []
    for method, calibration, decoding in TABLE2_ROWS:
        row = summary["methods"][method]
        fields = " & ".join(_f(row["per_field_mean"][f]) for f in FIELDS)
        body.append(
            f"{method} & {calibration} & {decoding} & "
            f"{_pm(row['weighted_macro_f1_mean'], row['weighted_macro_f1_std'])} & "
            f"{fields} & {_f(row['tuple_exact_match_mean'])} & "
            f"{_f(row['invalid_tuple_rate_mean'] * 100, 1)} \\\\"
        )
    header = ("ID & Calibration & Decoding & Weighted F1 & PS & VT & ES & EQ "
              "& Tuple Acc. & Invalid \\%")
    return _tabular("llrrrrrrrr", header, body)


def render_table3(regimes) -> str:
    body = []
    for label, row in regimes.items():
        ci = f"[{_f(row['ci_low'])}, {_f(row['ci_high'])}]"
        body.append(
            f"{label} & {_f(row['same_document'])} & "
            f"{_f(row['document_disjoint'])} & {_f(row['delta'])} & {ci} \\\\"
        )
    header = "Method & Same-document & Document-disjoint & $\\Delta$ & 95\\% CI"
    return _tabular("lrrrr", header, body)


def write_tables(out_dir, audit, summaries, regimes, input_files) -> Path:
    """Write the three tabulars, their captions and the provenance manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = {
        "table1_dataset.tex": render_table1(audit),
        # The main table reports the document-disjoint protocol; the
        # same-document protocol reaches the paper through Table 3.
        "table2_main.tex": render_table2(summaries["pdf_group"]),
        "table3_regimes.tex": render_table3(regimes),
    }
    for name, content in rendered.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    for stem, caption in CAPTIONS.items():
        (out_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    inputs = [str(Path(p)) for p in input_files]
    checksums = {p: file_sha256(p) for p in inputs}
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "tables": {
            name: {
                "source_script": "analysis/tables.py",
                "input_files": inputs,
                "input_sha256": checksums,
            }
            for name in TABLE_FILES
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_dir
