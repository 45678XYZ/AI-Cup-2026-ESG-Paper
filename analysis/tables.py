r"""Emit contract-4 tables: tabular only, captions beside them, manifest on top.

Contract section 5 splits the responsibility deliberately: content is C's
(number correctness), layout is D's (the 8-page budget). So nothing here emits
``\begin{table}``, ``\caption`` or ``\label``, and only booktabs rules are
used. Column counts follow ``contracts/examples/tables/`` because that is what
D measured the page budget against.

Every cell is computed. The manifest records the sha256 of every input file so
any number in the paper can be traced back to the artifacts that produced it
(contract section 5). "Every input file" means the ones a table's numbers were
actually computed from, which differs per table: table 1 is a dataset audit,
tables 2 and 3 are scored from per-row predictions. Checksumming anything else
yields a manifest that stays identical while an edited input moves every score.
"""

import json
from pathlib import Path

from analysis.audit import SPLITS_DIR
from analysis.load import METHODS, predictions_path
from paper.data import TEST_PATH, TRAIN_PATH, VAL_PATH, file_sha256
from paper.labels import FIELDS
from paper.provenance import git_sha, now_iso
from paper.train_config import PROTOCOLS, SEEDS

CONTRACT_VERSION = "1.0"

TABLE_FILES = ("table1_dataset.tex", "table2_main.tex", "table3_regimes.tex")

# The script that computed the numbers, not the one that formatted them.
SOURCE_SCRIPTS = {
    "table1_dataset.tex": "analysis/audit.py",
    "table2_main.tex": "analysis/aggregate.py",
    "table3_regimes.tex": "analysis/aggregate.py",
}

# The competition test split ships no labels, so its label-derived cells cannot
# be filled. Printing a development number there would be a fabricated cell.
NA = "n/a"

TABLE2_ROWS = (
    ("M0", "None", "Independent"), ("M1", "None", "Projection"),
    ("M2", "Global", "Projection"), ("M3", "Conditional", "Projection"),
    ("M4", "None", "17-state"), ("M5", "Global", "17-state"),
    ("M6", "Conditional", "17-state"),
)

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n) -> str:
    """Small counts spelled out, as academic prose wants them."""
    return _NUMBER_WORDS.get(n, f"{n:,}")


def _times(n) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{_word(n)} times")


def table_inputs(predictions_root, protocols=PROTOCOLS, seeds=SEEDS,
                 methods=METHODS) -> dict:
    """The files each table's numbers actually come from.

    Table 1 is a dataset audit, so it reads the dataset and the split
    manifests. Tables 2 and 3 are scored from per-row predictions -- nothing in
    ``analysis/`` reads ``results/*.json`` at all, so recording those would
    describe an audit trail that does not exist: an edited prediction file
    moves every score while the recorded checksums stay identical.
    """
    audited = [TRAIN_PATH, VAL_PATH, TEST_PATH, *sorted(SPLITS_DIR.glob("*.json"))]
    scored = {
        protocol: [predictions_path(protocol, seed, method, predictions_root)
                   for seed in seeds for method in methods]
        for protocol in protocols
    }
    return {
        # The main table reports the document-disjoint protocol; the
        # same-document protocol reaches the paper through table 3.
        "table1_dataset.tex": audited,
        "table2_main.tex": scored["pdf_group"],
        "table3_regimes.tex": [p for protocol in protocols for p in scored[protocol]],
    }


def build_captions(audit, seeds=SEEDS) -> dict:
    """Captions computed from the audit rather than stored as text.

    Captions are contract-4 deliverables and reach the paper verbatim, so the
    plan's "所有數字由 script 生成，不手抄" rule covers them exactly as it
    covers the figure's counts. A stored string is correct only until the next
    resample: the audit moves and the caption stays behind, saying something
    that was true last week.
    """
    dev = audit["development"]
    absent = audit["splits"]["calibration_without_misleading"]
    misleading = audit["splits"]["misleading_rows"]
    n_reports = len({row["pdf_url"] for row in misleading})
    return {
        "table1_dataset": (
            "Dataset and split statistics, regenerated from the versioned data "
            f"by analysis/audit.py. Misleading occurs {_times(len(misleading))} in the "
            f"whole development set, in {_word(n_reports)} different reports, and is "
            f"absent from the Calibration partition of {absent['n_without']} of the "
            f"{absent['n_rotations']} rotations. The competition test split ships "
            "no labels, so its label-derived cells are marked n/a."
        ),
        "table2_main": (
            "Controlled comparison of decision rules on identical base "
            "probabilities and identical test rows. Each cell is the mean over "
            f"{_word(len(seeds))} seeds of a single score computed on all "
            f"{dev['paragraphs']:,} concatenated test rows; $\\pm$ is the sample "
            "standard deviation across seeds, which reflects the variability of "
            "the whole pipeline -- fold assignment and training together -- "
            "rather than model stability."
        ),
        "table3_regimes": (
            "Same-document versus document-disjoint evaluation. The left column "
            "measures seen-report, unseen-paragraph generalisation and matches "
            "the competition distribution, since test and development draw on "
            f"the same {dev['pdfs']} reports; the right column measures "
            "generalisation to entirely unseen reports. $\\Delta$ is the gap "
            "between two estimation targets and is not a bias estimate."
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


def write_tables(out_dir, audit, summaries, regimes, inputs_by_table,
                 seeds=SEEDS) -> Path:
    """Write the three tabulars, their captions and the provenance manifest.

    ``inputs_by_table`` maps each table file to the inputs its numbers were
    computed from -- see ``table_inputs``. An empty list is refused rather than
    written: the manifest is the claim-evidence audit trail, and an empty one
    asserts nothing while looking exactly like a complete one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in TABLE_FILES:
        if not inputs_by_table.get(name):
            raise ValueError(
                f"no input files recorded for {name}. The manifest is the "
                "claim-evidence audit trail (contract §5); writing an empty "
                "one would let every number in the paper trace back to nothing."
            )

    rendered = {
        "table1_dataset.tex": render_table1(audit),
        # The main table reports the document-disjoint protocol; the
        # same-document protocol reaches the paper through Table 3.
        "table2_main.tex": render_table2(summaries["pdf_group"]),
        "table3_regimes.tex": render_table3(regimes),
    }
    for name, content in rendered.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    for stem, caption in build_captions(audit, seeds=seeds).items():
        (out_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    def entry(name):
        paths = [str(Path(p)) for p in inputs_by_table[name]]
        return {
            "source_script": SOURCE_SCRIPTS[name],
            "input_files": paths,
            "input_sha256": {p: file_sha256(p) for p in paths},
        }

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "tables": {name: entry(name) for name in TABLE_FILES},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_dir
