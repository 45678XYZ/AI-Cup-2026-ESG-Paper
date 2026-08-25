"""Does the Chinese mechanism hold in the four external corpora?

The frozen study's finding was not "projection scores better" -- it does not --
but an account of *why* the official metric cannot see legality: projection's
only lever on a child field is writing ``N/A``, so its gains land on one class
and its losses on the others, and a macro average over classes cancels them.

That account is mechanical, so it should replicate wherever the label schema
does. These tests check it on all five corpora at once, which is also the only
place the schema's own arithmetic is compared across them -- macro-F1 averages
over the classes *present in gold*, and that set is not the same everywhere.
"""

import numpy as np
import pytest

from analysis.multilingual_mechanism import (
    ARMS,
    CORPORA,
    build_report,
    corpus_order,
    gold_classes_present,
    leverage,
    load_corpus_arm,
)
from paper.labels import FIELD_WEIGHTS, FIELDS


@pytest.fixture(scope="module")
def report():
    return build_report(seeds=(42,))


def test_every_corpus_loads_without_the_undistributed_korean_text():
    """The Korean page text is not redistributed, so a loader that reached for
    it would make the whole analysis unrunnable from a clone. Gold comes from
    the split's canonical row order and the committed predictions instead."""
    for corpus in CORPORA:
        gold, pred = load_corpus_arm(corpus, ARMS[corpus], "pdf_group", 42, "M0")
        assert gold.shape == pred.shape
        assert gold.shape[1] == len(FIELDS)
        assert len(gold) == len(corpus_order(corpus))


def test_leverage_counts_only_the_classes_the_metric_averages_over():
    """macro-F1 is taken over classes present in gold, so a corpus missing one
    divides that field's weight by a smaller number. Korean has no Misleading
    row, which makes a point of evidence_quality F1 worth more there than
    anywhere else -- the same metric, a different exchange rate."""
    zh_gold, _ = load_corpus_arm("aicup_zh", ARMS["aicup_zh"], "pdf_group", 42, "M0")
    ko_gold, _ = load_corpus_arm("mlpromise_ko", ARMS["mlpromise_ko"],
                                 "pdf_group", 42, "M0")

    assert len(gold_classes_present(zh_gold)["evidence_quality"]) == 4
    assert len(gold_classes_present(ko_gold)["evidence_quality"]) == 3

    w = FIELD_WEIGHTS["evidence_quality"]
    assert leverage(zh_gold)["evidence_quality"] == pytest.approx(w / 4)
    assert leverage(ko_gold)["evidence_quality"] == pytest.approx(w / 3)


def test_the_leverage_table_is_not_the_same_in_every_corpus():
    """If it were, the metric's incentive structure would be a property of the
    task and could be stated once. It is a property of each corpus's gold."""
    tables = {c: leverage(load_corpus_arm(c, ARMS[c], "pdf_group", 42, "M0")[0])
              for c in CORPORA}
    distinct = {tuple(round(t[f], 6) for f in FIELDS) for t in tables.values()}
    assert len(distinct) > 1, tables


def test_the_ledger_direction_replicates_in_every_corpus(report):
    """The mechanism itself: projecting onto the legal states repairs N/A and
    damages substantive classes. Direction, not magnitude -- the corpora differ
    by a factor of five in size."""
    for corpus, entry in report["corpora"].items():
        ledger = entry["projection_ledger"]
        assert ledger["na"]["net"] > 0, corpus
        assert ledger["substantive"]["net"] < 0, corpus


def test_projection_removes_every_illegal_tuple_in_every_corpus(report):
    for corpus, entry in report["corpora"].items():
        assert entry["invalid_rate"]["M0"] > 0, corpus
        assert entry["invalid_rate"]["M1"] == 0.0, corpus


def test_the_preview_renders_the_multilingual_table():
    """The preview PDF is what a reader without TeX sees. A table absent from
    it is a table nobody checks."""
    from analysis.multilingual_mechanism import TABLE_NAME
    from analysis.tables import ALL_TABLE_FILES

    assert f"{TABLE_NAME}.tex" in ALL_TABLE_FILES


def test_the_report_records_which_arm_each_corpus_was_read_from(report):
    """Five corpora, five different backbones available. A report that did not
    name the arm would invite reading the numbers as a language comparison."""
    for corpus, entry in report["corpora"].items():
        assert entry["arm"] == ARMS[corpus]
        assert entry["n_rows"] > 0
