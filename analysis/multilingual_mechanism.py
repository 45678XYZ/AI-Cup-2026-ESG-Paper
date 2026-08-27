"""The frozen study's mechanism, re-measured on all five corpora.

The Chinese result was not that hierarchy projection scores better -- it does
not -- but an account of why the official metric cannot see legality. That
account is arithmetic rather than linguistic, so it should hold wherever the
label schema does, and the four ML-Promise corpora are the first chance to
check it against text the study never saw.

Two things are measured here that the per-language summaries do not report:

* **The repair ledger.** Projection's only lever on a child field is writing
  ``N/A``, so its gains land on one class and its losses on the others. A
  macro average over classes weights them equally, which is where a real gain
  in whole-tuple correctness goes missing. Direction is the claim; magnitude
  varies with corpus size.

* **The leverage table.** ``weight / |classes|`` is what a point of one class's
  F1 is worth in the weighted total. It is usually quoted as a property of the
  task, but macro-F1 averages over the classes *present in gold*, and that set
  differs between corpora -- Korean has no ``Misleading`` row at all. So the
  metric's exchange rate is a property of each corpus, not of the schema.

Scored from the committed per-row predictions against the split's canonical
row order. Nothing here loads the Korean page text, which is not redistributed:
the analysis has to run from a clone.

    python -m analysis.multilingual_mechanism
"""

import argparse
import json

import numpy as np

from analysis.cases import repair_ledger_by_class
from analysis.english_replication import _invalid_rate
from analysis.load import load_aligned
from analysis.metrics import tuple_accuracy, weighted_macro_f1
from paper.corpus import splits_dir
from paper.data import REPO_ROOT
from paper.labels import EVAL_FIELDS, FIELD_WEIGHTS, FIELDS, ID2LABEL

REPORT_VERSION = "1.0"

CORPORA = (
    "aicup_zh",
    "mlpromise_en",
    "mlpromise_fr",
    "mlpromise_ja",
    "mlpromise_ko",
)

# One arm per corpus, chosen before any of this was computed and named here so
# a reader cannot mistake the table for a language ranking. Each is that
# corpus's primary backbone at lambda = 0 -- the standard recipe, no structural
# objective -- because the mechanism is a property of the decision rule and
# adding a training-time constraint would confound it.
ARMS = {
    "aicup_zh": ".",
    "mlpromise_en": "runs_en/roberta_large/lambda_0.0",
    "mlpromise_fr": "runs_fr/xlm_roberta_large/lambda_0.0",
    "mlpromise_ja": "runs_ja/xlm_roberta_large/lambda_0.0",
    "mlpromise_ko": "runs_ko/xlm_roberta_large/lambda_0.0",
}

BACKBONES = {
    "aicup_zh": "Chinese RoBERTa-wwm-ext-large",
    "mlpromise_en": "RoBERTa-large",
    "mlpromise_fr": "XLM-R-large",
    "mlpromise_ja": "XLM-R-large",
    "mlpromise_ko": "XLM-R-large",
}

SEEDS = (42, 123, 456)
BASELINE, CONSTRAINED, DECODER = "M0", "M1", "M4"


def corpus_order(corpus: str) -> list[str]:
    """The row order the corpus's predictions are aligned to."""
    path = REPO_ROOT / splits_dir(corpus) / "pdf_group_seed42.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["canonical_row_order"]


def load_corpus_arm(corpus, arm, protocol, seed, method):
    """``(gold, pred)`` for one corpus, arm, protocol, seed and method."""
    return load_aligned(protocol, seed, method, corpus_order(corpus),
                        REPO_ROOT / arm)


def gold_classes_present(gold) -> dict:
    """Which classes of each field the corpus's gold actually contains.

    This is the set macro-F1 averages over -- the present-labels-only
    convention the competition's own scorer uses. A class with no gold row is
    not scored, so it is also not in the denominator.
    """
    gold = np.asarray(gold)
    return {
        field: sorted(ID2LABEL[field][int(c)] for c in np.unique(gold[:, j]))
        for j, field in enumerate(FIELDS)
    }


def leverage(gold) -> dict:
    """What a point of one class's F1 is worth in the weighted total.

    ``weight / |classes present|``. Raising one class's F1 by ``x`` raises the
    field's macro-F1 by ``x / |classes|`` and the weighted total by
    ``weight * x / |classes|``, so this is the exchange rate a participant is
    actually offered -- and it moves with the corpus, not just the schema.
    """
    present = gold_classes_present(gold)
    return {field: FIELD_WEIGHTS[field] / len(present[field]) for field in FIELDS}


def _metrics(gold, pred) -> dict:
    return {
        "weighted_macro_f1": weighted_macro_f1(gold, pred),
        "tuple_accuracy": tuple_accuracy(gold, pred),
    }


def corpus_mechanism(corpus, arm=None, seeds=SEEDS, protocol="pdf_group") -> dict:
    """Ledger, leverage and both metrics for one corpus, averaged over seeds."""
    arm = ARMS[corpus] if arm is None else arm
    ledger = {"na": {"repaired": 0, "destroyed": 0, "net": 0},
              "substantive": {"repaired": 0, "destroyed": 0, "net": 0}}
    by_class: dict[str, dict] = {}
    scores = {m: [] for m in (BASELINE, CONSTRAINED, DECODER)}
    invalid = {m: [] for m in (BASELINE, CONSTRAINED, DECODER)}
    gold = None

    for seed in seeds:
        loaded = {m: load_corpus_arm(corpus, arm, protocol, seed, m)
                  for m in (BASELINE, CONSTRAINED, DECODER)}
        gold = loaded[BASELINE][0]
        for method, (g, pred) in loaded.items():
            scores[method].append(_metrics(g, pred))
            invalid[method].append(_invalid_rate(pred))

        seed_ledger = repair_ledger_by_class(
            gold, loaded[BASELINE][1], loaded[CONSTRAINED][1])
        for bucket in ("na", "substantive"):
            for key in ("repaired", "destroyed"):
                ledger[bucket][key] += seed_ledger[bucket][key]
        for name, entry in seed_ledger["by_class"].items():
            acc = by_class.setdefault(name, {"repaired": 0, "destroyed": 0})
            acc["repaired"] += entry["repaired"]
            acc["destroyed"] += entry["destroyed"]

    for bucket in ledger.values():
        bucket["net"] = bucket["repaired"] - bucket["destroyed"]
    for entry in by_class.values():
        entry["net"] = entry["repaired"] - entry["destroyed"]

    present = gold_classes_present(gold)
    return {
        "corpus": corpus,
        "arm": arm,
        "backbone": BACKBONES[corpus],
        "protocol": protocol,
        "seeds": list(seeds),
        "n_rows": int(len(gold)),
        "classes_present": {f: len(present[f]) for f in FIELDS},
        "classes_absent": {
            f: sorted(set(EVAL_FIELDS[f]) - set(present[f])) for f in FIELDS
        },
        "leverage": leverage(gold),
        "projection_ledger": {**ledger, "by_class": by_class},
        "metrics": {
            m: {k: float(np.mean([s[k] for s in scores[m]]))
                for k in ("weighted_macro_f1", "tuple_accuracy")}
            for m in scores
        },
        "invalid_rate": {m: float(np.mean(invalid[m])) for m in invalid},
    }


def build_report(corpora=CORPORA, seeds=SEEDS) -> dict:
    entries = {c: corpus_mechanism(c, seeds=seeds) for c in corpora}
    leverages = {
        c: tuple(round(e["leverage"][f], 9) for f in FIELDS)
        for c, e in entries.items()
    }
    return {
        "report_version": REPORT_VERSION,
        "corpora": entries,
        "observations": {
            "ledger_direction_holds_everywhere": all(
                e["projection_ledger"]["na"]["net"] > 0
                and e["projection_ledger"]["substantive"]["net"] < 0
                for e in entries.values()
            ),
            "projection_removes_all_illegal_tuples": all(
                e["invalid_rate"][CONSTRAINED] == 0.0 for e in entries.values()
            ),
            "distinct_leverage_tables": len(set(leverages.values())),
            "corpora_missing_a_class": sorted(
                c for c, e in entries.items()
                if any(e["classes_absent"][f] for f in FIELDS)
            ),
        },
    }


LABELS = {
    "aicup_zh": "Chinese (AI CUP)",
    "mlpromise_en": "English",
    "mlpromise_fr": "French",
    "mlpromise_ja": "Japanese",
    "mlpromise_ko": "Korean",
}

TABLE_NAME = "table7_multilingual_mechanism"


def render_table(report) -> str:
    """One row per corpus: what projection repairs, what it costs, what each
    metric then reports. The two delta columns are the whole table -- they are
    the same rows scored two ways."""
    body = []
    for corpus, e in report["corpora"].items():
        led, met = e["projection_ledger"], e["metrics"]
        d_f1 = met[CONSTRAINED]["weighted_macro_f1"] - met[BASELINE]["weighted_macro_f1"]
        d_row = met[CONSTRAINED]["tuple_accuracy"] - met[BASELINE]["tuple_accuracy"]
        body.append(
            f"{LABELS[corpus]} & {e['n_rows']:,} & "
            f"{e['invalid_rate'][BASELINE] * 100:.2f} & "
            f"{led['na']['net']:+d} & {led['substantive']['net']:+d} & "
            f"{d_f1:+.4f} & \\textbf{{{d_row:+.4f}}} \\\\"
        )
    header = ("Corpus & Rows & M0 illegal \\% & $N\\!/\\!A$ net & Subst.\\ net & "
              "$\\Delta$ official & $\\Delta$ whole-row")
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n")


def build_caption(report) -> str:
    entries = report["corpora"]
    obs = report["observations"]
    n = len(entries)
    missing = obs["corpora_missing_a_class"]

    parts = [
        f"Projection (M1) versus independent argmax (M0) on {_word(n)} corpora, "
        f"each using its preselected primary backbone at $\\lambda=0$ and "
        f"document-disjoint evaluation; values are means over "
        f"{_word(len(SEEDS))} seeds.",
        f"M1 guarantees 0\\% invalid output. The $N\\!/\\!A$ and substantive "
        f"net columns summarize changes in correct child predictions after "
        f"projection.",
    ]
    if missing:
        names = ", ".join(LABELS[c] for c in missing)
        absent = {c: entries[c]["classes_absent"] for c in missing}
        _, classes = next(
            (f, v) for f, v in absent[missing[0]].items() if v)
        parts.append(
            f"Official-score deltas should not be compared directly across "
            f"corpora because macro-F1 averages over gold-present classes, and "
            f"{names} contains no {classes[0]} example."
        )
    return " ".join(parts)


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven"}


def _word(n) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def write_report(out_dir=None) -> dict:
    out_dir = REPO_ROOT / "tables" if out_dir is None else out_dir
    report = build_report()
    with open(out_dir / "multilingual_mechanism.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    (out_dir / f"{TABLE_NAME}.tex").write_text(render_table(report), encoding="utf-8")
    (out_dir / f"{TABLE_NAME}_caption.txt").write_text(
        build_caption(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    from pathlib import Path
    report = write_report(Path(args.out_dir) if args.out_dir else None)
    obs = report["observations"]
    print(f"mechanism -> tables/{TABLE_NAME}.tex")
    print(f"  ledger direction holds in every corpus: "
          f"{obs['ledger_direction_holds_everywhere']}")
    print(f"  distinct leverage tables across {len(report['corpora'])} corpora: "
          f"{obs['distinct_leverage_tables']}")


if __name__ == "__main__":
    main()
