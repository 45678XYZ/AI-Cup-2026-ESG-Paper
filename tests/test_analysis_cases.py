"""The qualitative half of C's W3: what the invalid tuples actually are.

Table 2 says structured decoding barely moves weighted macro-F1 while removing
every invalid tuple. That is a puzzle on its face, and the numbers here are
what resolve it: which hierarchy rule the model breaks, and whether projection
repairs a field or destroys it. Both are counted, never described.
"""

import numpy as np
import pytest

from analysis.cases import (
    misleading_cases,
    parent_overrides,
    case_analysis,
    projection_ledger,
    violation_breakdown,
)
from analysis.load import EXAMPLES_ROOT, METHODS, load_aligned
from paper.data import canonical_row_order, load_dev
from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID

DEV = load_dev()
ORDER = canonical_row_order(DEV)


def _row(ps, vt, es, eq):
    return [LABEL2ID["promise_status"][ps],
            LABEL2ID["verification_timeline"][vt],
            LABEL2ID["evidence_status"][es],
            LABEL2ID["evidence_quality"][eq]]


def test_violation_breakdown_names_the_rule_each_row_breaks():
    pred = np.array([
        _row("No", "already", "N/A", "N/A"),      # PS=No 卻給了時程
        _row("Yes", "already", "No", "Clear"),    # ES=No 卻評了品質
        _row("Yes", "already", "Yes", "N/A"),     # ES=Yes 卻沒評品質
        _row("Yes", "N/A", "Yes", "Clear"),       # PS=Yes 卻缺時程
        _row("No", "N/A", "N/A", "N/A"),          # 合法
    ])
    out = violation_breakdown(pred)
    assert out["n_invalid"] == 4
    assert out["by_rule"]["promise_no_children_set"] == 1
    assert out["by_rule"]["evidence_no_quality_set"] == 1
    assert out["by_rule"]["evidence_yes_quality_missing"] == 1
    assert out["by_rule"]["promise_yes_children_absent"] == 1
    assert sum(out["by_rule"].values()) == out["n_invalid"]


def test_violation_breakdown_reports_nothing_for_legal_predictions():
    pred = np.array([_row("No", "N/A", "N/A", "N/A")])
    out = violation_breakdown(pred)
    assert out["n_invalid"] == 0
    assert out["invalid_rate"] == 0.0


def test_projection_ledger_separates_repairs_from_damage():
    """A field the rule changes is repaired, destroyed, or was wrong either way.
    Collapsing the three hides exactly what Table 2 needs explained."""
    gold   = np.array([_row("Yes", "already", "Yes", "Clear")])
    before = np.array([_row("Yes", "already", "No", "Clear")])   # 非法
    after  = np.array([_row("Yes", "already", "No", "N/A")])     # 投影：EQ 改壞了
    led = projection_ledger(gold, before, after)
    assert led["fields_repaired"] == 0
    assert led["fields_destroyed"] == 1
    assert led["fields_wrong_either_way"] == 0
    assert led["net_fields"] == -1


def test_projection_ledger_counts_a_genuine_repair():
    gold   = np.array([_row("No", "N/A", "N/A", "N/A")])
    before = np.array([_row("No", "already", "N/A", "N/A")])
    after  = np.array([_row("No", "N/A", "N/A", "N/A")])
    led = projection_ledger(gold, before, after)
    assert led["fields_repaired"] == 1 and led["net_fields"] == 1


def test_case_analysis_runs_on_the_example_set():
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    assert out["protocol"] == "pdf_group" and out["seed"] == 42
    assert out["independent"]["n_invalid"] >= 0
    assert set(out["projection"]) >= {"fields_repaired", "fields_destroyed", "net_fields"}
    # 投影後必然合法 —— 這是結構保證，不是實驗結果
    assert out["after_projection"]["n_invalid"] == 0


def test_case_analysis_rejects_a_method_that_cannot_be_invalid():
    """M1-M6 are legal by construction; asking for their breakdown is a bug."""
    with pytest.raises(ValueError, match="M0"):
        case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV, baseline="M3")


def test_case_analysis_is_written_as_a_deliverable(tmp_path):
    """It has to land in a file: D writes the Discussion from it, and a number
    that only exists in a console is not evidence anyone can check."""
    from analysis.cases import write_case_analysis

    out = write_case_analysis(tmp_path, ORDER, EXAMPLES_ROOT, dev=DEV,
                              protocols=("pdf_group",), seeds=(42, 123))
    assert out.exists() and out.name == "case_analysis.json"

    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["runs"]) == 2
    assert doc["totals"]["n_invalid"] == sum(r["independent"]["n_invalid"]
                                             for r in doc["runs"])
    # 規則分布必須加總回非法列數，否則有列沒有被歸類
    assert sum(doc["totals"]["by_rule"].values()) == doc["totals"]["n_invalid"]


# ------------------------------------- where the repair lands, and what it costs
# The net figure hides the mechanism. Projection overwrites a child field with
# N/A whenever the parent forbids it, so its gains concentrate on the N/A
# classes and its losses fall on the substantive ones. macro-F1 weights every
# class equally, which is why a large gain in tuple accuracy can arrive with no
# movement in the official metric at all.


def test_ledger_by_class_separates_na_from_substantive():
    from analysis.cases import repair_ledger_by_class

    gold   = np.array([_row("No",  "N/A",     "N/A", "N/A"),
                       _row("Yes", "already", "Yes", "Clear")])
    before = np.array([_row("No",  "already", "N/A", "Clear"),   # 非法
                       _row("Yes", "already", "Yes", "Clear")])
    after  = np.array([_row("No",  "N/A",     "N/A", "N/A"),     # 投影修好
                       _row("Yes", "already", "Yes", "Clear")])

    led = repair_ledger_by_class(gold, before, after)
    assert led["by_class"]["verification_timeline.N/A"]["repaired"] == 1
    assert led["by_class"]["evidence_quality.N/A"]["repaired"] == 1
    assert led["na"]["repaired"] == 2 and led["na"]["destroyed"] == 0
    assert led["substantive"]["repaired"] == 0


def test_ledger_records_a_substantive_class_being_destroyed():
    from analysis.cases import repair_ledger_by_class

    gold   = np.array([_row("Yes", "already", "Yes", "Clear")])
    before = np.array([_row("No",  "already", "Yes", "Clear")])   # PS 錯，其餘對
    after  = np.array([_row("No",  "N/A",     "N/A", "N/A")])     # 投影把對的抹掉

    led = repair_ledger_by_class(gold, before, after)
    assert led["substantive"]["destroyed"] == 3
    assert led["na"]["repaired"] == 0
    assert led["net"] == -3


def test_unobserved_states_reports_what_gold_never_shows():
    from analysis.cases import unobserved_states

    gold = np.array([_row("No", "N/A", "N/A", "N/A")])
    pred = np.array([_row("Yes", "already", "Yes", "Clear")])
    out = unobserved_states(gold, pred)
    assert out["n_unobserved_in_gold"] == 16          # 17 個合法狀態中 gold 只出現 1 個
    assert out["n_emitted_unobserved"] == 1
    assert ("Yes", "already", "Yes", "Clear") in [tuple(s) for s in out["emitted_unobserved"]]


def test_case_analysis_carries_the_class_ledger_and_unobserved_states():
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    assert set(out["by_class"]) >= {"by_class", "na", "substantive", "net"}
    assert set(out["unobserved"]) == {"M4", "M5", "M6"}
    for rec in out["unobserved"].values():
        assert {"n_unobserved_in_gold", "n_emitted_unobserved"} <= set(rec)


def test_written_totals_aggregate_the_na_split(tmp_path):
    import json

    from analysis.cases import write_case_analysis
    out = write_case_analysis(tmp_path, ORDER, EXAMPLES_ROOT, dev=DEV,
                              protocols=("pdf_group",), seeds=(42, 123))
    doc = json.loads(out.read_text(encoding="utf-8"))
    t = doc["totals"]
    assert t["na"]["net"] + t["substantive"]["net"] == t["net_fields"]


def test_unobserved_states_are_checked_on_the_decoders_not_the_projection():
    """Projection can only ever emit a state its parent chain allows, so asking
    it about unseen states answers nothing. The question is about M4-M6, which
    search the whole legal space."""
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    assert set(out["unobserved"]) == {"M4", "M5", "M6"}
    for method, rec in out["unobserved"].items():
        assert "n_emitted_unobserved" in rec


def test_misleading_cases_records_every_gold_instance_per_method():
    """Plan section 4.5 permits exactly two treatments of a class with n=2: the
    per-instance record and a sensitivity score computed without it. This is
    the first one, and until now it was missing from every deliverable."""
    dev = load_dev()
    order = canonical_row_order(dev)
    rows = misleading_cases(order, dev, "pdf_group", 42, EXAMPLES_ROOT)
    gold_rows = [r for r in dev if r["evidence_quality"] == "Misleading"]
    assert len(rows) == len(gold_rows) == 2
    for row in rows:
        assert row["gold"]["evidence_quality"] == "Misleading"
        assert set(row["predicted"]) == set(METHODS)
        assert row["pdf_url"]
        # every prediction is a real label, not an index
        for pred in row["predicted"].values():
            assert pred in EVAL_FIELDS["evidence_quality"]


def test_case_analysis_carries_the_misleading_instances():
    dev = load_dev()
    order = canonical_row_order(dev)
    out = case_analysis("pdf_group", 42, order, EXAMPLES_ROOT, dev=dev)
    assert len(out["misleading_cases"]) == 2


# --- the mechanism behind M4-M1 --------------------------------------------
#
# The projection decides promise_status first and never reconsiders it. The
# 17-state decoder scores whole tuples, so a confident evidence_quality can
# overturn a marginal promise_status. That is the stated mechanism for the one
# contrast where decoding loses to projection -- and until this function it was
# an assertion with no number behind it.


def test_parent_overrides_counts_only_the_rows_the_decoder_revised():
    # row 0 unchanged; row 1 wrong -> correct; row 2 correct -> wrong.
    gold      = np.array([[0, 0, 0, 0], [1, 4, 2, 3], [0, 0, 0, 0]])
    projected = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    decoded   = np.array([[0, 0, 0, 0], [1, 4, 2, 3], [1, 4, 2, 3]])

    out = parent_overrides(gold, projected, decoded)
    assert out["field"] == "promise_status"
    assert out["n_rows"] == 3
    assert out["n_changed"] == 2
    assert out["to_correct"] == 1
    assert out["to_wrong"] == 1
    assert out["wrong_to_wrong"] == 0
    # the three buckets must partition the changed rows, or the ledger lies
    assert out["to_correct"] + out["to_wrong"] + out["wrong_to_wrong"] == out["n_changed"]


def test_parent_overrides_also_scores_the_whole_tuple_on_those_rows():
    """Whether overturning the parent helped is not answerable from the parent
    alone: the decoder rewrites the children to match."""
    gold      = np.array([[1, 4, 2, 3]])
    projected = np.array([[0, 0, 0, 0]])
    decoded   = np.array([[1, 4, 2, 3]])
    out = parent_overrides(gold, projected, decoded)
    assert out["tuple_correct_before"] == 0
    assert out["tuple_correct_after"] == 1


def test_case_analysis_carries_the_override_ledger():
    dev = load_dev()
    order = canonical_row_order(dev)
    out = case_analysis("pdf_group", 42, order, EXAMPLES_ROOT, dev=dev)
    ledger = out["parent_overrides"]
    assert ledger["n_rows"] == len(order)
    assert 0 <= ledger["n_changed"] <= ledger["n_rows"]


def test_case_analysis_records_what_the_metric_pays_for_an_illegal_row():
    """The report states that a structurally unusable answer still collects a
    substantial share of the weighted per-field credit. That sentence is the
    difference between "the metric ignores legality" and "the metric pays for
    breaking it", so the number behind it cannot live only in prose."""
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    credit = out["partial_credit_on_invalid"]
    assert 0.0 < credit < 1.0
    # It is a weighted mean of four per-field accuracies, so it cannot exceed 1.


def test_case_analysis_separates_what_the_hierarchy_decides_from_what_it_leaves_open():
    """The paper's structural argument rests on this split: the hierarchy
    determines only whether a child field is N/A, and the model already makes
    that call well, while the substantive choice -- where the headroom is -- is
    left entirely open. Two numbers per child field, and they must be reported
    together or the comparison disappears."""
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    info = out["hierarchy_information"]
    assert set(info) == {"verification_timeline", "evidence_status", "evidence_quality"}
    for field, row in info.items():
        assert 0.0 <= row["na_determination"] <= 1.0, field
        assert 0.0 <= row["substantive_choice"] <= 1.0, field


def test_the_two_rates_are_averaged_across_runs_not_summed():
    """Both are rates. Summing them across runs would give a number with no
    interpretation that still sits in the same table as the counts, which is
    exactly how a wrong figure survives review."""
    import json
    from analysis.cases import write_case_analysis

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = write_case_analysis(tmp, ORDER, EXAMPLES_ROOT, dev=DEV,
                                   protocols=("pdf_group",), seeds=(42, 123))
        totals = json.loads(path.read_text(encoding="utf-8"))["totals"]
    assert 0.0 < totals["partial_credit_on_invalid"] < 1.0
    for row in totals["hierarchy_information"].values():
        assert 0.0 <= row["na_determination"] <= 1.0
