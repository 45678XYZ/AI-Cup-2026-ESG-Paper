"""Synthetic example files for contract 3 (results) and contract 4 (tables).

    python -m contracts.make_examples

Contract 3 gets a full set of M0-M6 predictions plus their summaries, so C can
write and test the bootstrap, Holm correction, per-class and sensitivity code
today instead of waiting for the GPU. Contract 4 gets placeholder ``.tex`` so
D can lay out and measure the 8-page budget against the real column structure.

**The numbers are fabricated.** Every summary carries ``"synthetic": true`` and
every table cell is ``--``. They are shaped correctly and internally
consistent; they mean nothing. Their only job is to let downstream code run.

What is deliberately reproduced from the real thing, because it is what
downstream code has to cope with:

  * M0 emits hierarchy-invalid tuples; M1-M6 never do
  * ``Misleading`` has n=2, so its per-class F1 is degenerate
  * every method covers the same 2,000 rows, each appearing exactly once
"""

import argparse
import json
import random
from pathlib import Path

from paper.artifacts import CONTRACT_VERSION, prediction_row, write_predictions
from paper.data import REPO_ROOT, index_by_id, load_dev
from paper.evaluate import build_results, write_results
from paper.labels import EVAL_FIELDS, FIELDS, STATES
from paper.provenance import git_sha, now_iso
from paper.train_config import PROTOCOLS, SEEDS

METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

# Fabricated error rates, chosen only so the seven files are not identical.
# Do not read an ordering into the resulting scores: which method comes out on
# top here is an artefact of these constants, and the study exists precisely to
# answer that question with real data.
ERROR_RATE = {"M0": 0.30, "M1": 0.31, "M2": 0.29, "M3": 0.28,
              "M4": 0.30, "M5": 0.28, "M6": 0.27}


def _fake_predictions(rows, split, method, rng):
    """One method's predictions over all 2,000 rows, assembled across rotations."""
    by_id = index_by_id(rows)
    rate = ERROR_RATE[method]
    records = []

    for rot in split["rotations"]:
        for row_id in rot["test_ids"]:
            row = by_id[row_id]
            if method == "M0":
                # Independent per-field corruption: the fields fall out of
                # agreement, so some tuples end up hierarchy-invalid.
                pred = {
                    f: (rng.choice(EVAL_FIELDS[f]) if rng.random() < rate else row[f])
                    for f in FIELDS
                }
            else:
                # Corrupt to another *legal* state, so validity always holds.
                if rng.random() < rate:
                    s = rng.choice(STATES)
                    pred = dict(zip(FIELDS, (s.ps, s.vt, s.es, s.eq)))
                else:
                    pred = {f: row[f] for f in FIELDS}
            records.append(prediction_row(row, rot["k"], pred))

    return records


def make_contract3(split, rows, out_root, methods=METHODS):
    protocol, seed = split["protocol"], split["seed"]
    written = []

    for method in methods:
        rng = random.Random(f"{protocol}:{seed}:{method}")
        records = _fake_predictions(rows, split, method, rng)

        pred_path = Path(out_root) / "predictions" / f"{protocol}_seed{seed}_{method}.csv.gz"
        write_predictions(pred_path, records)

        # Same builder the real M0-M6 runner will use, so the envelope C
        # validates against here is the envelope C receives in W3.
        summary = build_results(
            records,
            protocol=protocol,
            seed=seed,
            method=method,
            predictions_path=pred_path,
            data_checksum=split["data_checksum"],
            synthetic=True,
            decision_params={
                str(rot["k"]): {
                    "calibration_biases": None if method in ("M0", "M1", "M4") else "<per-class biases>",
                    "fallback_applied": rot["calibration_absent_classes"],
                    "decoder": {"alpha": [1.0] * 4, "mode": "fixed"} if method in ("M4", "M5", "M6") else None,
                }
                for rot in split["rotations"]
            },
        )
        res_path = write_results(
            Path(out_root) / "results" / f"{protocol}_seed{seed}_{method}.json", summary,
        )

        written.append((pred_path, res_path))
        print(f"  {method}: weighted_macro_f1={summary['weighted_macro_f1']:.4f} "
              f"invalid={summary['invalid_tuple_rate']:.3%}")
    return written


# --------------------------------------------------------------------------
# Contract 4
# --------------------------------------------------------------------------

PLACEHOLDER = "--"

# Column set is plan §5 Table 2, including Tuple Acc.: D measures the 8-page
# budget against this file, so a column missing here is a column D discovers in
# W4 when the table no longer fits.
TABLE2 = r"""\begin{tabular}{llrrrrrrrr}
\toprule
ID & Calibration & Decoding & Weighted F1 & PS & VT & ES & EQ & Tuple Acc. & Invalid \%% \\
\midrule
%s
\bottomrule
\end{tabular}
"""

TABLE2_NUMERIC_COLUMNS = 7

TABLE2_ROWS = [
    ("M0", "None", "Independent"), ("M1", "None", "Projection"),
    ("M2", "Global", "Projection"), ("M3", "Conditional", "Projection"),
    ("M4", "None", "17-state"), ("M5", "Global", "17-state"),
    ("M6", "Conditional", "17-state"),
]

TABLE1 = r"""\begin{tabular}{lrr}
\toprule
Statistic & Development & Test \\
\midrule
Paragraphs & %(p)s & %(p)s \\
Source reports (PDFs) & %(p)s & %(p)s \\
Companies & %(p)s & %(p)s \\
Legal states observed & %(p)s & %(p)s \\
\midrule
\multicolumn{3}{l}{\emph{Rarest classes}} \\
\quad within\_2\_years & %(p)s & %(p)s \\
\quad Misleading & %(p)s & %(p)s \\
\bottomrule
\end{tabular}
""" % {"p": PLACEHOLDER}

TABLE3 = r"""\begin{tabular}{lrrrr}
\toprule
Method & Same-document & Document-disjoint & $\Delta$ & 95\%% CI \\
\midrule
M0 & %(p)s & %(p)s & %(p)s & %(p)s \\
Best calibrated projection & %(p)s & %(p)s & %(p)s & %(p)s \\
Best valid-state decoder & %(p)s & %(p)s & %(p)s & %(p)s \\
\bottomrule
\end{tabular}
""" % {"p": PLACEHOLDER}

CAPTIONS = {
    "table1_dataset": "Dataset and split statistics. Misleading occurs only twice in the "
                      "development data and is absent from most calibration partitions.",
    "table2_main": "Controlled comparison of decision rules on identical base probabilities "
                   "and identical test rows. Cells are seed mean$\\pm$std.",
    "table3_regimes": "Same-document versus document-disjoint evaluation. $\\Delta$ is the gap "
                      "between two estimation targets, not a bias estimate.",
}


def make_contract4(out_root):
    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    body = "\n".join(
        f"{mid} & {cal} & {dec} & " + " & ".join([PLACEHOLDER] * TABLE2_NUMERIC_COLUMNS) + r" \\"
        for mid, cal, dec in TABLE2_ROWS
    )
    files = {
        "table1_dataset.tex": TABLE1,
        "table2_main.tex": TABLE2 % body,
        "table3_regimes.tex": TABLE3,
    }
    for name, content in files.items():
        (tables_dir / name).write_text(content, encoding="utf-8")
    for stem, caption in CAPTIONS.items():
        (tables_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "synthetic": True,
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "tables": {
            name: {"source_script": "contracts/make_examples.py", "input_files": [],
                   "input_sha256": {}}
            for name in files
        },
    }
    with open(tables_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"  wrote {len(files)} tabular placeholders + captions + manifest")
    return tables_dir


def main():
    ap = argparse.ArgumentParser(description="Generate contract 3 and 4 example files.")
    # Defaults cover the full 2 x 3 x 7 = 42 files contract §4.1 specifies.
    # A single (protocol, seed) would leave C unable to exercise the two paths
    # that only exist across them: the 3-seed mean+/-std of Table 2, and the
    # same-document vs document-disjoint comparison of Table 3.
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "contracts" / "examples")
    args = ap.parse_args()

    rows = load_dev()
    n = 0
    for protocol in args.protocols:
        for seed in args.seeds:
            with open(REPO_ROOT / "splits" / f"{protocol}_seed{seed}.json", encoding="utf-8") as f:
                split = json.load(f)
            print(f"contract 3 — {protocol} seed{seed}:")
            n += len(make_contract3(split, rows, args.out))
    print("contract 4 (tables):")
    make_contract4(args.out)
    print(f"\n{n} predictions + {n} results under {args.out}")
    print("\nEVERY NUMBER ABOVE IS FABRICATED. The files are shaped correctly and")
    print("internally consistent so downstream code can run; the scores and their")
    print("ordering mean nothing and must never reach a table.")


if __name__ == "__main__":
    main()
